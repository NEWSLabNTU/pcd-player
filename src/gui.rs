use crate::data::{BBoxPvrcnn, InfoPoint, VoxAnn};
use anyhow::{bail, Result};
use itertools::Itertools;
use kiss3d::{
    camera::{ArcBall, Camera},
    event::{Action, Key, WindowEvent},
    planar_camera::PlanarCamera,
    post_processing::PostProcessingEffect,
    text::Font,
    window::{State, Window},
};
use kiss3d_utils::WindowPlotExt;
use nalgebra::{Isometry3, Point2, Point3, Vector2, Vector3};
use palette::{FromColor, Hsv, Srgb};
use pcd_rs::Field;
use std::{
    fs,
    path::{Path, PathBuf},
    rc::Rc,
};
use supervisely_format as sv;

pub struct App {
    pcd_files: Vec<PathBuf>,
    ann_dir: Option<PathBuf>,
    vox_dir: Option<PathBuf>,
    pcd_dir: PathBuf,
    index: usize,
    state: StateKind,
    camera: ArcBall,
    font: Rc<Font>,
    cache: Option<Cache>,
    colored: bool,
}

impl State for App {
    fn step(&mut self, window: &mut Window) {
        let result = self.try_step(window);
        if let Err(err) = result {
            eprintln!("Error: {err}");
            window.close();
        }
    }

    fn cameras_and_effect(
        &mut self,
    ) -> (
        Option<&mut dyn Camera>,
        Option<&mut dyn PlanarCamera>,
        Option<&mut dyn PostProcessingEffect>,
    ) {
        (Some(&mut self.camera), None, None)
    }
}

impl App {
    pub fn build<P1, P2, P3>(
        pcd_dir: P1,
        ann_dir: Option<P2>,
        vox_dir: Option<P3>,
        colored: bool,
    ) -> Result<Self>
    where
        P1: AsRef<Path>,
        P2: AsRef<Path>,
        P3: AsRef<Path>,
    {
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

                let Some(ext) = path.extension() else {
                    return Ok(None);
                };
                if ext != "pcd" {
                    return Ok(None);
                }

                Ok(Some(path))
            })
            .filter_map(|path| path.transpose())
            .try_collect()?;
        pcd_files.sort_unstable();

        let ann_dir = ann_dir.map(|dir| dir.as_ref().to_path_buf());
        let vox_dir = vox_dir.map(|dir| dir.as_ref().to_path_buf());
        let pcd_dir = pcd_dir.as_ref().to_path_buf();

        let eye = Point3::from([0.0f32, -80.0, 32.0]);
        let at = Point3::origin();
        let mut camera = ArcBall::new(eye, at);
        camera.set_up_axis(Vector3::from([0.0, 0.0, 1.0]));

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
            font: Font::default(),
        })
    }

    fn try_step(&mut self, window: &mut Window) -> Result<()> {
        self.update(window)?;
        self.render(window)?;
        Ok(())
    }

    fn update(&mut self, window: &mut Window) -> Result<()> {
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
            &Point2::from([5.0; 2]),
            40.0,
            &Font::default(),
            &Point3::from([0.0, 204.0, 0.0]),
        );

        self.draw_pcd(window)?;
        if self.ann_dir.is_some() {
            self.draw_ann(window)?;
        }
        if self.vox_dir.is_some() {
            self.draw_voxel(window)?;
        }

        window.draw_axes(Point3::origin(), 10.0);

        Ok(())
    }

    fn file_stem(&self) -> &str {
        self.pcd_files[self.index]
            .file_stem()
            .unwrap()
            .to_str()
            .unwrap()
    }

    fn get_ann_file(&self) -> Option<PathBuf> {
        let dir = self.ann_dir.as_ref()?;
        let path = dir.join(format!("{}.pcd.json", &self.file_stem()));
        let ok = path.is_file() || path.is_symlink();
        ok.then_some(path)
    }

    fn get_bbox_file(&self) -> Option<PathBuf> {
        let path = self.pcd_dir.join(format!("{}.json", &self.file_stem()));
        let ok = path.is_file() || path.is_symlink();
        ok.then_some(path)
    }

    fn get_vox_file(&self) -> Option<PathBuf> {
        let dir = self.vox_dir.as_ref()?;
        let path = dir.join(format!("{}.json", &self.file_stem()));
        let ok = path.is_file() || path.is_symlink();
        ok.then_some(path)
    }

    fn get_pcd_file(&self) -> &PathBuf {
        &self.pcd_files[self.index]
    }

    fn load_pcd(&mut self) -> Result<&Cache> {
        let cache = if self.cache.as_ref().unwrap().index != self.index {
            self.update_cache()?
        } else if self.cache.is_some() {
            self.cache.as_ref().unwrap()
        } else {
            self.update_cache()?
        };
        Ok(cache)
    }

    fn update_cache(&mut self) -> Result<&Cache> {
        let pcd_file = self.get_pcd_file();
        let bbox_file = self.get_bbox_file();
        let vox_file = self.get_vox_file();

        let cache = Cache::load_from_files(self.index, pcd_file, bbox_file, vox_file)?;
        let cache = self.cache.insert(cache);
        Ok(cache)
    }

    fn draw_pcd(&mut self, window: &mut Window) -> Result<()> {
        let default_color = Point3::from([1.0; 3]);
        // for point in pcd_rs::DynReader::open(self.get_pcd_file())? {
        //     let point = Point3::from_slice(&point?.xyz().unwrap());
        //     window.draw_point(&point, &color);
        // }
        let colored = self.colored;
        let cache = self.load_pcd()?;

        cache.points.iter().for_each(|point| {
            let point3 = &point.point;
            let color = if colored {
                match point.device_id {
                    Some(device_id) => get_color(device_id, point.active),
                    None => default_color,
                }
            } else {
                default_color
            };
            window.draw_point(point3, &color)
        });

        let bbox_info = &cache.bbox_info;
        if let Some(bbox_info) = bbox_info {
            if let Some(symmetric_score) = bbox_info.symmetric_score {
                window.draw_text(
                    &format!(
                        "Box: dx = {:.2}, \
                         dy = {:.2}, \
                         dz = {:.2}, \
                         symmetrical_score = {:.3}",
                        bbox_info.dx, bbox_info.dy, bbox_info.dz, symmetric_score
                    ),
                    &Point2::from([5.0, 55.0]),
                    40.0,
                    &Font::default(),
                    &Point3::from([0.0, 204.0, 0.0]),
                );
            }
        }
        Ok(())
    }

    fn draw_voxel(&mut self, window: &mut Window) -> Result<()> {
        let vox_ann = self.load_pcd()?.vox_ann.clone();
        let color = Point3::from([1_f32, 0.0, 0.0]);

        let Some(vox_ann) = vox_ann else {
            return Ok(());
        };

        vox_ann.vox_boxes.iter().for_each(|vox_box| {
            let size = Point3::from([
                vox_box.size_x as f32,
                vox_box.size_y as f32,
                vox_box.size_z as f32,
            ]);
            let translation = Vector3::from([
                vox_box.center_x as f32,
                vox_box.center_y as f32,
                vox_box.center_z as f32,
            ]);
            let axis_angle = match vox_box.heading {
                Some(heading) => Vector3::from([0 as f32, 0 as f32, heading as f32]),
                _ => Vector3::from([0 as f32, 0 as f32, 0 as f32]),
            };
            let pose = Isometry3::new(translation, axis_angle);
            let custom_color = match vox_box.color {
                Some(custom_color) => Point3::from([
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
                    &text,
                    &Point3::cast(pos),
                    50.0,
                    &Point3::new(255.0, 255.0, 255.0),
                    &self.camera,
                    &self.font,
                );
            }
        });

        Ok(())
    }

    fn draw_ann(&self, window: &mut Window) -> Result<()> {
        let ann_file = self.get_ann_file().expect("Unable to load Annotation file");
        let sv::PointCloudAnnotation { figures, .. } =
            serde_json::from_str(&fs::read_to_string(ann_file)?)?;

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
                let (r, g, b) = Srgb::from_color(Hsv::new(hue, 1.0, 1.0)).into_components();
                Point3::from([r, g, b])
            };

            let size = Point3::from([
                dimensions.x as f32,
                dimensions.y as f32,
                dimensions.z as f32,
            ]);
            let translation =
                Vector3::from([position.x as f32, position.y as f32, position.z as f32]);
            let axis_angle =
                Vector3::from([rotation.x as f32, rotation.y as f32, rotation.z as f32]);
            let pose = Isometry3::new(translation, axis_angle);

            window.draw_box(size, pose, color)
        }

        Ok(())
    }
}

#[derive(Debug)]
struct Cache {
    index: usize,
    points: Vec<InfoPoint>,
    bbox_info: Option<BBoxPvrcnn>,
    vox_ann: Option<VoxAnn>,
}

impl Cache {
    pub fn load_from_files(
        index: usize,
        pcd_file: impl AsRef<Path>,
        bbox_file: Option<impl AsRef<Path>>,
        vox_file: Option<impl AsRef<Path>>,
    ) -> Result<Self> {
        let reader = pcd_rs::DynReader::open(pcd_file)?;
        let points = reader
            .map(|point| {
                let point = point?;
                let point3 = Point3::from(point.clone().to_xyz().unwrap());
                let fields = &point.0;

                let device_id = match fields.get(5) {
                    Some(field) => {
                        let Field::I32(value) = field else {
                            bail!("invalid point type");
                        };
                        Some(value[0] as u64)
                    }
                    None => None,
                };
                let active = match fields.get(6) {
                    Some(field) => {
                        let Field::I32(value) = field else {
                            bail!("invalid point type");
                        };
                        Some(value[0] as u64)
                    }
                    None => None,
                };

                Ok(InfoPoint {
                    point: point3,
                    device_id,
                    active,
                })
            })
            .try_collect()?;
        let bbox_info: Option<BBoxPvrcnn> = match bbox_file {
            Some(bbox_file) => {
                // Read the JSON contents of the file as an instance of `User`.
                let bbox_info = serde_json::from_str(&fs::read_to_string(bbox_file)?)?;
                Some(bbox_info)
            }
            None => None,
        };
        let vox_ann: Option<VoxAnn> = match vox_file {
            Some(vox_file) => {
                // Read the JSON contents of the file as an instance of `User`.
                let vox_ann = serde_json::from_str(&fs::read_to_string(vox_file)?)?;
                Some(vox_ann)
            }
            None => None,
        };

        Ok(Cache {
            index,
            points,
            bbox_info,
            vox_ann,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum StateKind {
    Pause,
    Playing,
}

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
    text: &str,
    pos: &Point3<f32>,
    scale: f32,
    color: &Point3<f32>,
    camera: &dyn Camera,
    font: &Rc<Font>,
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
