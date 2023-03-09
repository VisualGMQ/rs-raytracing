use std::io::Write;

use crate::math;

pub struct Image {
    data: Vec<math::Color>,
    width: u32,
    height: u32,
}

impl Image {
    pub fn new(width: u32, height: u32) -> Self {
        Self { data: vec![math::Color::zero(); (width * height) as usize], width, height }
    }

    pub fn set(&mut self, x: u32, y: u32, color: &math::Color) {
        self.data[(x + y * self.width) as usize] = *color;
    }

    pub fn get(&self, x: u32, y: u32) -> math::Color {
        self.data[(x + y * self.width) as usize]
    }

    pub fn width(&self) -> u32 {
        self.width
    }

    pub fn height(&self) -> u32 {
        self.height
    }

    pub fn save2ppm(&self, filename: &str) -> std::io::Result<()> {
        let mut writer = std::io::BufWriter::new(std::fs::File::create(filename)?);
        writer.write_all(b"P3\r\n")?;
        writer.write_all(format!("{} {}\r\n", self.width(), self.height()).as_bytes())?;
        writer.write_all(b"255\r\n")?;

        for y in 0..self.height() {
            for x in 0..self.width() {
                let color = self.get(x, y);
                let color_str = format!("{} {} {}  ", (color.x * 255.0) as u32, (color.y * 255.0) as u32, (color.z * 255.0) as u32);
                writer.write_all(color_str.as_bytes())?;
            }
            writer.write_all(b"\r\n")?;
        }
        writer.flush()?;
        Ok(())
    }
}
