use anyhow::Result;
use clap::Parser;
use itertools::Itertools;
use kiss3d::{
    camera::{ArcBall, Camera},
    event::{Action, Key, WindowEvent},
    light::Light,
    text::Font,
    window::Window,
};
use kiss3d_utils::WindowPlotExt;
use nalgebra::{Isometry3, Point2, Point3, Translation3, UnitQuaternion, Vector2, Vector3};
use num_traits::cast::FromPrimitive;
use palette::FromColor;
use pcd_rs::Field;
use serde::{Deserialize, Serialize};
use std::{
    io::BufReader,
    path::{Path, PathBuf},
};
use supervisely_format as sv;

#[derive(Parser)]
struct Opts {
    #[clap(long)]
    pub pcd_dir: PathBuf,

    #[clap(long)]
    pub ann_dir: Option<PathBuf>,
    #[clap(long)]
    pub vox_dir: Option<PathBuf>,
    #[clap(long)]
    pub colored: bool,
    #[clap(long, default_value = "1.0")]
    pub point_size: f32,
}

#[derive(Debug, Clone, PartialEq)]
struct InfoPoint {
    pub point: Point3<f32>,
    pub device_id: Option<u64>,
    pub active: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct BBox3D {
    pub center_x: f64,
    pub center_y: f64,
    pub center_z: f64,
    pub size_x: f64,
    pub size_y: f64,
    pub size_z: f64,
    pub color: Option<(f64, f64, f64)>,
    pub heading: Option<f64>,
    pub id: Option<usize>,
}

impl BBox3D {
    pub fn center(&self) -> Point3<f64> {
        let Self {
            center_x,
            center_y,
            center_z,
            ..
        } = *self;
        Point3::new(center_x, center_y, center_z)
    }
    pub fn vertex(
        &self,
        x_choice: bool,
        y_choice: bool,
        z_choice: bool,
        heading: f64,
    ) -> Point3<f64> {
        let rotation = UnitQuaternion::from_euler_angles(0.0, 0.0, heading);
        let translation = Translation3::new(self.center_x, self.center_y, self.center_z);
        let pose = Isometry3::from_parts(translation, rotation);
        let point = {
            let x = self.size_x / 2.0 * if x_choice { 1.0 } else { -1.0 };
            let y = self.size_y / 2.0 * if y_choice { 1.0 } else { -1.0 };
            let z = self.size_z / 2.0 * if z_choice { 1.0 } else { -1.0 };
            Point3::new(x, y, z)
        };
        pose * point
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct VoxAnn {
    pub vox_boxes: Vec<BBox3D>,
}

fn main() -> Result<()> {
    let Opts {
        pcd_dir,
        ann_dir,
        vox_dir,
        colored,
        point_size,
    } = Opts::parse();

    let mut window = Window::new("pcd-player");

    window.set_light(Light::StickToCamera);
    window.set_point_size(point_size);

    let mut state = App::build(pcd_dir, ann_dir, vox_dir, colored)?;

    // window.render_loop(state);

    while !window.should_close() {
        state.try_step(&mut window)?;
        state.render(&mut window)?;
    }

    Ok(())
}

#[derive(Debug)]
struct App {
    pcd_files: Vec<PathBuf>,
    ann_dir: Option<PathBuf>,
    vox_dir: Option<PathBuf>,
    pcd_dir: PathBuf,
    index: usize,
    state: StateKind,
    camera: ArcBall,
    // camera: FirstPerson,
    cache: Option<Cache>,
    colored: bool,
}

#[derive(Debug)]
struct Cache {
    index: usize,
    points: Vec<InfoPoint>,
    bbox_info: Option<BBoxPvrcnn>,
    vox_ann: Option<VoxAnn>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum StateKind {
    Pause,
    Playing,
}

// impl State for App {
//     fn step(&mut self, window: &mut Window) {
//         let result = self.try_step(window);
//         if result.is_err() {
//             window.close();
//         }
//     }
// }

fn get_color(device_id: u64, active: Option<u64>) -> Point3<f32> {
    let color = match device_id {
        1 => Point3::from([1.0, 0.5, 0.0]),
        2 => Point3::from([0.0, 1.0, 0.0]),
        3 => Point3::from([0.0, 1.0, 1.0]),
        _ => Point3::from([0.5, 0.5, 0.5]),
    };
    if let Some(active) = active {
        if active == 0 {
            return Point3::from([0.5, 0.5, 0.5]);
        }
    }
    color
}

fn draw_text_3d(
    window: &mut Window,
    camera: &dyn kiss3d::camera::Camera,
    text: &str,
    pos: &Point3<f32>,
    scale: f32,
    font: &std::rc::Rc<kiss3d::text::Font>,
    color: &Point3<f32>,
) {
    let window_size = Vector2::new(window.size()[0] as f32, window.size()[1] as f32);
    let mut window_coord = camera.project(pos, &window_size);
    if window_coord.x.is_nan() || window_coord.y.is_nan() {
        return;
    }
    window_coord.y = window_size.y - window_coord.y;
    if window_coord.x >= window_size.x
        || window_coord.x < 0.0
        || window_coord.y >= window_size.y
        || window_coord.y < 0.0
    {
        return;
    }
    let coord: &Point2<f32> = &(window_coord * 2.0).into();
    window.draw_text(text, coord, scale, font, color);
}

impl App {
    fn build(
        pcd_dir: impl AsRef<Path>,
        ann_dir: Option<impl AsRef<Path>>,
        vox_dir: Option<impl AsRef<Path>>,
        colored: bool,
    ) -> Result<Self> {
        let mut pcd_files: Vec<_> = pcd_dir
            .as_ref()
            .read_dir()?
            .map(|entry| -> Result<_> {
                let entry = entry?;
                let file_type = entry.file_type()?;

                if !(file_type.is_file() || file_type.is_symlink()) {
                    return Ok(None);
                }

                let path = entry.path();
                if path.extension().and_then(|ext| ext.to_str()) != Some("pcd") {
                    return Ok(None);
                }

                Ok(Some(path))
            })
            .filter_map(|path| path.transpose())
            .try_collect()?;
        pcd_files.sort_unstable();

        let ann_dir = ann_dir.map(|dir| PathBuf::from(dir.as_ref()));
        let vox_dir = vox_dir.map(|dir| PathBuf::from(dir.as_ref()));

        let pcd_dir = PathBuf::from(pcd_dir.as_ref());

        let eye = Point3::from_slice(&[0.0f32, -80.0, 32.0]);
        let at = Point3::origin();
        let mut camera = ArcBall::new(eye, at);
        camera.set_up_axis(Vector3::from_column_slice(&[0., 0., 1.]));

        Ok(Self {
            pcd_files,
            ann_dir,
            vox_dir,
            pcd_dir,
            index: 0,
            state: StateKind::Pause,
            camera,
            cache: None,
            colored,
        })
    }

    fn file_stem(&self) -> String {
        self.pcd_files[self.index]
            .file_stem()
            .unwrap()
            .to_str()
            .unwrap()
            .into()
    }

    fn get_ann_file(&self) -> PathBuf {
        if let Some(dir) = &self.ann_dir {
            let ann_file_path = dir.join(format!("{}.pcd.json", &self.file_stem()));
            if !ann_file_path.exists() {
                panic!("{} doest not exist!", ann_file_path.display());
            } else {
                ann_file_path
            }
        } else {
            panic!("Annotation directory is not specified.")
        }
    }

    fn get_bbox_file(&self) -> Option<PathBuf> {
        let bbox_file_path = self.pcd_dir.join(format!("{}.json", &self.file_stem()));
        if !bbox_file_path.exists() {
            None
        } else {
            Some(bbox_file_path)
        }
    }

    fn get_vox_file(&self) -> Option<PathBuf> {
        if let Some(vox_dir) = &self.vox_dir {
            let vox_file_path = vox_dir.join(format!("{}.json", &self.file_stem()));
            if !vox_file_path.exists() {
                None
            } else {
                Some(vox_file_path)
            }
        } else {
            None
        }
    }

    fn get_pcd_file(&self) -> &PathBuf {
        &self.pcd_files[self.index]
    }

    fn load_pcd(&mut self) -> Result<(&Vec<InfoPoint>, Option<BBoxPvrcnn>, Option<VoxAnn>)> {
        if self.cache.is_none() || (self.cache.as_ref().unwrap().index != self.index) {
            let points = pcd_rs::DynReader::open(self.get_pcd_file())?
                .filter_map(Result::ok)
                .map(|point| {
                    let point3 = Point3::from_slice(&point.clone().to_xyz().unwrap());
                    let (device_id, active) = if point.0.len() >= 6 {
                        let device_id = match &point.clone().0[5] {
                            Field::I8(device_id) => u64::from_i8(device_id[0]),
                            Field::I16(device_id) => u64::from_i16(device_id[0]),
                            Field::I32(device_id) => u64::from_i32(device_id[0]),
                            Field::U8(device_id) => u64::from_u8(device_id[0]),
                            Field::U16(device_id) => u64::from_u16(device_id[0]),
                            Field::U32(device_id) => u64::from_u32(device_id[0]),
                            Field::F32(device_id) => u64::from_f32(device_id[0]),
                            Field::F64(device_id) => u64::from_f64(device_id[0]),
                        };
                        let active = if point.0.len() >= 7 {
                            match &point.clone().0[6] {
                                Field::I8(active) => u64::from_i8(active[0]),
                                Field::I16(active) => u64::from_i16(active[0]),
                                Field::I32(active) => u64::from_i32(active[0]),
                                Field::U8(active) => u64::from_u8(active[0]),
                                Field::U16(active) => u64::from_u16(active[0]),
                                Field::U32(active) => u64::from_u32(active[0]),
                                Field::F32(active) => u64::from_f32(active[0]),
                                Field::F64(active) => u64::from_f64(active[0]),
                            }
                        } else {
                            None
                        };

                        (device_id, active)
                    } else {
                        (None, None)
                    };
                    InfoPoint {
                        point: point3,
                        device_id,
                        active,
                    }
                })
                .collect();
            let bbox_file = self.get_bbox_file();
            let bbox_info: Option<BBoxPvrcnn> = match bbox_file {
                Some(bbox_file) => {
                    let file = std::fs::File::open(bbox_file)?;
                    let reader = BufReader::new(file);

                    // Read the JSON contents of the file as an instance of `User`.
                    let bbox_info = serde_json::from_reader(reader)?;
                    Some(bbox_info)
                }
                _ => None,
            };
            let vox_file = self.get_vox_file();
            let vox_ann: Option<VoxAnn> = match vox_file {
                Some(vox_file) => {
                    let file = std::fs::File::open(vox_file)?;
                    let reader = BufReader::new(file);

                    // Read the JSON contents of the file as an instance of `User`.
                    let vox_ann = serde_json::from_reader(reader)?;
                    Some(vox_ann)
                }
                _ => None,
            };

            self.cache = Some(Cache {
                index: self.index,
                points,
                bbox_info,
                vox_ann,
            });
        }

        Ok((
            &self.cache.as_ref().unwrap().points,
            self.cache.as_ref().unwrap().bbox_info.clone(),
            self.cache.as_ref().unwrap().vox_ann.clone(),
        ))

        // if let Some(cache) = &self.cache {
        //     let Cache { index, points } = cache;
        //     if *index == self.index {
        //         return Ok(points);
        //     }
        // }

        // todo!()
        // Ok(&self.cache.as_ref().unwrap().points)
        // Ok(&self.cache.unwrap().points)
    }

    fn draw_pcd(&mut self, window: &mut Window) -> Result<()> {
        let default_color = Point3::from([1.0; 3]);
        // for point in pcd_rs::DynReader::open(self.get_pcd_file())? {
        //     let point = Point3::from_slice(&point?.xyz().unwrap());
        //     window.draw_point(&point, &color);
        // }
        let colored = self.colored;
        self.load_pcd()?.0.iter().for_each(|point| {
            let point3 = &point.point;
            let color = match colored {
                true => match point.device_id {
                    Some(device_id) => get_color(device_id, point.active),
                    _ => default_color,
                },
                false => default_color,
            };
            window.draw_point(point3, &color)
        });
        let bbox_info = self.load_pcd()?.1;
        if let Some(bbox_info) = bbox_info {
            if let Some(symmetric_score) = bbox_info.symmetric_score {
                window.draw_text(
                    &format!(
                        "Box: dx = {:.2}, dy = {:.2}, dz = {:.2}, symmetrical_score = {:.3}",
                        bbox_info.dx, bbox_info.dy, bbox_info.dz, symmetric_score
                    ),
                    &Point2::from_slice(&[5.0, 55.0]),
                    40.0,
                    &Font::default(),
                    &Point3::from_slice(&[0.0, 204.0, 0.0]),
                );
            }
        }
        Ok(())
    }

    fn draw_voxel(&mut self, window: &mut Window) -> Result<()> {
        let vox_ann = self.load_pcd()?.2;
        let color = Point3::from_slice(&[1_f32, 0.0, 0.0]);
        if let Some(vox_ann) = vox_ann {
            vox_ann.vox_boxes.iter().for_each(|vox_box| {
                let size = Point3::from_slice(&[
                    vox_box.size_x as f32,
                    vox_box.size_y as f32,
                    vox_box.size_z as f32,
                ]);
                let translation = Vector3::from_column_slice(&[
                    vox_box.center_x as f32,
                    vox_box.center_y as f32,
                    vox_box.center_z as f32,
                ]);
                let axis_angle = match vox_box.heading {
                    Some(heading) => {
                        Vector3::from_column_slice(&[0 as f32, 0 as f32, heading as f32])
                    }
                    _ => Vector3::from_column_slice(&[0 as f32, 0 as f32, 0 as f32]),
                };
                let pose = Isometry3::new(translation, axis_angle);
                let custom_color = match vox_box.color {
                    Some(custom_color) => Point3::from_slice(&[
                        custom_color.0 as f32,
                        custom_color.1 as f32,
                        custom_color.2 as f32,
                    ]),
                    _ => color,
                };
                window.draw_box(size, pose, custom_color);
                if let Some(id) = vox_box.id {
                    let text = format!("{id}");
                    let pos = match vox_box.heading {
                        Some(heading) => vox_box.vertex(true, true, true, heading),
                        _ => vox_box.center(),
                    };
                    draw_text_3d(
                        window,
                        &self.camera,
                        &text,
                        &Point3::cast(pos),
                        50.0,
                        &kiss3d::text::Font::default(),
                        &Point3::new(255.0, 255.0, 255.0),
                    );
                }
            });
        }

        Ok(())
    }

    fn draw_ann(&self, window: &mut Window) -> Result<()> {
        let sv::PointCloudAnnotation { figures, .. } =
            serde_json::from_str(&std::fs::read_to_string(self.get_ann_file())?)?;

        for fig in figures {
            let sv::PointCloudFigure {
                object_key,
                geometry:
                    sv::PointCloudGeometry {
                        position,
                        rotation,
                        dimensions,
                    },
                ..
            } = fig;

            // TODO: Use better choices of color
            let color = {
                let hue = (u128::from_str_radix(&object_key, 16)? % 360) as f32;
                let (r, g, b) =
                    palette::Srgb::from_color(palette::Hsv::new(hue, 1.0, 1.0)).into_components();
                Point3::from_slice(&[r, g, b])
            };

            let size = Point3::from_slice(&[
                dimensions.x as f32,
                dimensions.y as f32,
                dimensions.z as f32,
            ]);
            let translation = Vector3::from_column_slice(&[
                position.x as f32,
                position.y as f32,
                position.z as f32,
            ]);
            let axis_angle = Vector3::from_column_slice(&[
                rotation.x as f32,
                rotation.y as f32,
                rotation.z as f32,
            ]);
            let pose = Isometry3::new(translation, axis_angle);

            window.draw_box(size, pose, color)
        }

        Ok(())
    }

    fn try_step(&mut self, window: &mut Window) -> Result<()> {
        use StateKind::*;

        window.events().iter().for_each(|event| {
            use Action as A;
            use Key as K;
            use WindowEvent as E;

            match event.value {
                E::Key(K::Space, A::Press, _) => {
                    // toggle play/pause
                    self.state = match self.state {
                        Pause => Playing,
                        Playing => Pause,
                    };
                }
                E::Key(K::P, A::Press, _) => {
                    // previous frame
                    if self.state == Pause {
                        self.index = if self.index > 0 { self.index - 1 } else { 0 };
                    }
                }
                E::Key(K::N, A::Press, _) => {
                    // next frame
                    if self.state == Pause {
                        self.index = if self.index + 1 < self.pcd_files.len() {
                            self.index + 1
                        } else {
                            self.index
                        };
                    }
                }
                E::Key(K::Left, A::Release, _) => {
                    let curr_yaw = self.camera.yaw();
                    self.camera.set_yaw(curr_yaw - 0.05);
                }
                E::Key(K::Right, A::Release, _) => {
                    let curr_yaw = self.camera.yaw();
                    self.camera.set_yaw(curr_yaw + 0.05);
                }
                E::Key(K::Down, A::Release, _) => {
                    self.camera.set_pitch(self.camera.pitch() - 0.05);
                }
                E::Key(K::Up, A::Release, _) => {
                    self.camera.set_pitch(self.camera.pitch() + 0.05);
                }
                E::Key(K::R, A::Press, _) => {
                    // restart
                    self.state = Pause;
                    self.index = 0;
                }
                _ => {}
            }
        });

        match self.state {
            Pause => {
                self.render(window)?;
            }
            Playing => {
                let new_index = self.index + 1;
                if new_index < self.pcd_files.len() {
                    self.index = new_index;
                } else {
                    self.state = Pause;
                }

                self.render(window)?;
            }
        }

        Ok(())
    }

    fn render(&mut self, window: &mut Window) -> Result<()> {
        window.draw_text(
            &format!("Index: {}, name: {}", self.index, self.file_stem()),
            &Point2::from_slice(&[5.0; 2]),
            40.0,
            &Font::default(),
            &Point3::from_slice(&[0.0, 204.0, 0.0]),
        );

        self.draw_pcd(window)?;
        if self.ann_dir.is_some() {
            self.draw_ann(window)?;
        }
        if self.vox_dir.is_some() {
            self.draw_voxel(window)?;
        }

        window.draw_axes(Point3::origin(), 10.0);

        window.render_with_camera(&mut self.camera);
        println!(
            "Position: {}, Yaw: {}, Pitch: {}",
            self.camera.eye(),
            self.camera.yaw(),
            self.camera.pitch()
        );
        // println!("Position: {}", self.camera.eye());

        Ok(())
    }
}

#[derive(Debug, Clone, Deserialize)]
struct BBoxPvrcnn {
    // pub x: f64,
    // pub y: f64,
    // pub z: f64,
    pub dx: f64,
    pub dy: f64,
    pub dz: f64,
    pub symmetric_score: Option<f64>,
}
