mod data;
mod gui;

use anyhow::Result;
use clap::Parser;
use gui::App;
use kiss3d::{light::Light, window::Window};
use std::path::PathBuf;

/// The point cloud sequence player.
#[derive(Parser)]
struct Opts {
    /// The directory that contains .pcd point cloud files.
    #[clap(long)]
    pub pcd_dir: PathBuf,

    /// The directory that contains annotation files.
    #[clap(long)]
    pub ann_dir: Option<PathBuf>,

    /// The directory that contains voxel files.
    #[clap(long)]
    pub vox_dir: Option<PathBuf>,

    /// Enable point coloring.
    #[clap(long)]
    pub colored: bool,

    /// Set the plotted point size.
    #[clap(long, default_value = "1.0")]
    pub point_size: f32,
}

fn main() -> Result<()> {
    let Opts {
        pcd_dir,
        ann_dir,
        vox_dir,
        colored,
        point_size,
    } = Opts::parse();

    let mut window = Window::new(env!("CARGO_BIN_NAME"));

    window.set_light(Light::StickToCamera);
    window.set_point_size(point_size);

    let state = App::build(pcd_dir, ann_dir, vox_dir, colored)?;
    window.render_loop(state);

    Ok(())
}
