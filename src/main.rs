use rs_raytracing::{ppm::Image, math::Color};

fn main() {
    let mut img = Image::new(255, 255);
    for x in 0..img.width() {
        for y in 0..img.height() {
            img.set(x, y, &Color::new(x as f32 / img.width() as f32, y as f32 / img.height() as f32, 0.0, 1.0));
        }
    }
    img.save2ppm("./test.ppm").unwrap();
}
