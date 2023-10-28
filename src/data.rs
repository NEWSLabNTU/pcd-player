use nalgebra::{Isometry3, Point3, Translation3, UnitQuaternion};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Deserialize)]
pub struct BBoxPvrcnn {
    // pub x: f64,
    // pub y: f64,
    // pub z: f64,
    pub dx: f64,
    pub dy: f64,
    pub dz: f64,
    pub symmetric_score: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VoxAnn {
    pub vox_boxes: Vec<BBox3D>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct InfoPoint {
    pub point: Point3<f32>,
    pub device_id: Option<u64>,
    pub active: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BBox3D {
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
