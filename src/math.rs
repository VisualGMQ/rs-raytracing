use std::default::Default;
use std::f32::consts::PI;
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

pub const PI_DIV_2: f32 = std::f32::consts::FRAC_PI_2;
pub const PI_DIV_4: f32 = std::f32::consts::FRAC_PI_4;
pub const PI2: f32 = PI * 2.0;
pub const PI_INV: f32 = 1.0 / PI;

macro_rules! declare_vec_op {
    ($name:ident, $type:ty, $triat_name:ident, $func_name:ident, $op:tt, $($mem:ident),+) => {
        impl $triat_name for $name {
            type Output = Self;

            fn $func_name(self, rhs: Self) -> Self::Output {
                $name {
                    $(
                        $mem: self.$mem $op rhs.$mem,
                    )+
                }
            }
        }
    };
}

macro_rules! declare_vec_op_assign {
    ($name:ident, $type:ty, $triat_name:ident, $func_name:ident, $op:tt, $($mem:ident),+) => {
        impl $triat_name for $name {
            fn $func_name(&mut self, rhs: Self) {
                $(
                    self.$mem $op rhs.$mem;
                )+
            }
        }
    };
}

macro_rules! declare_vec {
    ($name:ident, $type:ty, $($mem:ident),+) => {
        #[derive(Debug, PartialEq, Copy, Clone, Default)]
        pub struct $name {
            $(
                pub $mem : $type,
            )+
        }

        impl $name {
            pub fn new($( $mem: $type,)+) -> $name {
                $name {
                    $( $mem, )+
                }
            }

            pub fn zero() -> $name {
                $name {
                    $( $mem: 0.0, )+
                }
            }

            pub fn length_square(&self) -> f32 {
                $(
                    self.$mem * self.$mem +
                )+
                0.0
            }

            pub fn length(&self) -> f32 {
                self.length_square().sqrt()
            }

            pub fn normalize(&self) -> $name {
                *self / self.length()
            }

            pub fn dot(&self, rhs: &$name) -> f32 {
                $(
                    self.$mem * rhs.$mem +
                )+
                0.0
            }
        }

        declare_vec_op!($name, $type, Add, add, + $(,$mem)+ );
        declare_vec_op!($name, $type, Sub, sub, - $(,$mem)+ );
        declare_vec_op!($name, $type, Mul, mul, * $(,$mem)+ );
        declare_vec_op!($name, $type, Div, div, / $(,$mem)+ );

        impl Neg for $name {
            type Output = Self;

            fn neg(self) -> Self::Output {
                Self::new(
                    $(
                        -self.$mem,
                    )+
                )
            }
        }


        impl Mul<f32> for $name {
            type Output = $name;

            fn mul(self, rhs: f32) -> Self::Output {
                $name {
                    $(
                        $mem: self.$mem * rhs,
                    )+
                }
            }
        }

        impl Mul<$name> for f32 {
            type Output = $name;

            fn mul(self, rhs: $name) -> Self::Output {
                rhs * self
            }
        }

        impl Div<f32> for $name {
            type Output = $name;

            fn div(self, rhs: f32) -> Self::Output {
                $name {
                    $(
                        $mem: self.$mem / rhs,
                    )+
                }
            }
        }

        declare_vec_op_assign!($name, $type, AddAssign, add_assign, += $(,$mem)+ );
        declare_vec_op_assign!($name, $type, SubAssign, sub_assign, -= $(,$mem)+ );
        declare_vec_op_assign!($name, $type, MulAssign, mul_assign, *= $(,$mem)+ );
        declare_vec_op_assign!($name, $type, DivAssign, div_assign, /= $(,$mem)+ );


        impl MulAssign<f32> for $name {
            fn mul_assign(&mut self, rhs: f32) {
                self.x *= rhs;
                self.y *= rhs;
            }
        }

        impl DivAssign<f32> for $name {
            fn div_assign(&mut self, rhs: f32) {
                self.x /= rhs;
                self.y /= rhs;
            }
        }
    };
}

declare_vec!(Vec2, f32, x, y);
declare_vec!(Vec3, f32, x, y, z);
declare_vec!(Vec4, f32, x, y, z, w);

impl Vec2 {
    pub fn cross(&self, rhs: &Vec2) -> f32 {
        self.x * rhs.y - self.y * rhs.x
    }
}

impl Vec3 {
    pub fn cross(&self, rhs: &Vec3) -> Vec3 {
        Vec3 {
            x: self.y * rhs.z - self.z * rhs.y,
            y: self.z * rhs.x - self.x * rhs.z,
            z: self.x * rhs.y - self.y * rhs.x,
        }
    }
}

// row-major matrix

macro_rules! declare_mat {
    ($name:ident, $dim:expr) => {
        #[derive(Debug, Clone, Copy)]
        pub struct $name {
            data: [f32; $dim * $dim],
        }

        impl $name {
            pub fn from_row(data: &[f32; $dim * $dim]) -> $name {
                $name { data: data.clone() }
            }

            pub fn from_col(data: &[f32; $dim * $dim]) -> $name {
                let mut mat = $name::zeros();
                for x in 0..$dim {
                    for y in 0..$dim {
                        mat.set(x, y, data[y + $dim * x]);
                    }
                }
                mat
            }

            pub fn zeros() -> $name {
                $name {
                    data: [0.; $dim * $dim],
                }
            }

            pub fn ones() -> $name {
                $name {
                    data: [1.; $dim * $dim],
                }
            }

            pub fn identity() -> $name {
                let mut result = $name::zeros();
                for i in 0..$dim {
                    result.set(i, i, 1.0);
                }
                result
            }

            pub fn get(&self, x: usize, y: usize) -> f32 {
                self.data[x + y * $dim]
            }

            pub fn set(&mut self, x: usize, y: usize, value: f32) {
                self.data[x + y * $dim] = value;
            }
        }

        impl Mul for $name {
            type Output = Self;

            fn mul(self, rhs: Self) -> Self::Output {
                let mut result = $name {
                    data: [0.0; $dim * $dim],
                };
                for i in 0..$dim {
                    for j in 0..$dim {
                        let mut sum = 0.0;
                        for k in 0..$dim {
                            sum += self.get(k, i) * rhs.get(j, k);
                        }
                        result.set(j, i, sum);
                    }
                }
                result
            }
        }

        impl Mul<f32> for $name {
            type Output = Self;

            fn mul(self, rhs: f32) -> Self::Output {
                let mut result = $name::zeros();
                for x in 0..$dim {
                    for y in 0..$dim {
                        result.set(x, y, self.get(x, y) * rhs);
                    }
                }
                result
            }
        }

        impl Div<f32> for $name {
            type Output = Self;

            fn div(self, rhs: f32) -> Self::Output {
                self * (1.0 / rhs)
            }
        }

        impl PartialEq for $name {
            fn eq(&self, other: &Self) -> bool {
                self.data == other.data
            }
        }
    };
}

declare_mat!(Mat2, 2);
declare_mat!(Mat3, 3);
declare_mat!(Mat4, 4);

impl Mul<Vec2> for Mat2 {
    type Output = Vec2;

    fn mul(self, rhs: Vec2) -> Self::Output {
        Vec2::new(
            self.get(0, 0) * rhs.x + self.get(1, 0) * rhs.y,
            self.get(0, 1) * rhs.x + self.get(1, 1) * rhs.y,
        )
    }
}

impl Mul<Vec3> for Mat3 {
    type Output = Vec3;

    fn mul(self, rhs: Vec3) -> Self::Output {
        Vec3::new(
            self.get(0, 0) * rhs.x + self.get(1, 0) * rhs.y + self.get(2, 0) * rhs.z,
            self.get(0, 1) * rhs.x + self.get(1, 1) * rhs.y + self.get(2, 1) * rhs.z,
            self.get(0, 2) * rhs.x + self.get(1, 2) * rhs.y + self.get(2, 2) * rhs.z,
        )
    }
}

impl Mul<Vec4> for Mat4 {
    type Output = Vec4;

    fn mul(self, rhs: Vec4) -> Self::Output {
        Vec4::new(
            self.get(0, 0) * rhs.x
                + self.get(1, 0) * rhs.y
                + self.get(2, 0) * rhs.z
                + self.get(3, 0) * rhs.w,
            self.get(0, 1) * rhs.x
                + self.get(1, 1) * rhs.y
                + self.get(2, 1) * rhs.z
                + self.get(3, 1) * rhs.w,
            self.get(0, 2) * rhs.x
                + self.get(1, 2) * rhs.y
                + self.get(2, 2) * rhs.z
                + self.get(3, 2) * rhs.w,
            self.get(0, 3) * rhs.x
                + self.get(1, 3) * rhs.y
                + self.get(2, 3) * rhs.z
                + self.get(3, 3) * rhs.w,
        )
    }
}

impl Mat2 {
    pub fn det(&self) -> f32 {
        self.get(0, 0) * self.get(1, 1) - self.get(1, 0) * self.get(0, 1)
    }

    pub fn inverse(&self) -> Option<Self> {
        let d = self.det();
        if d.abs() <= f32::EPSILON {
            return None;
        }
        Some(*self / d)
    }
}

impl Mat3 {
    pub fn det(&self) -> f32 {
        self.get(0, 0) * self.get(1, 1) * self.get(2, 2)
            + self.get(2, 0) * self.get(0, 1) * self.get(1, 2)
            + self.get(1, 0) * self.get(2, 1) * self.get(0, 2)
            - (self.get(2, 0) * self.get(1, 1) * self.get(0, 2)
                + self.get(1, 0) * self.get(0, 1) * self.get(2, 2)
                + self.get(0, 0) * self.get(1, 2) * self.get(2, 1))
    }

    pub fn inverse(&self) -> Option<Self> {
        let d = self.det();
        if d.abs() <= f32::EPSILON {
            return None;
        }
        Some(*self / d)
    }
}

// coordination

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct Polar {
    pub theta: f32, // in radians
    pub length: f32,
}

impl Polar {
    pub fn new(theta: f32, length: f32) -> Self {
        Polar { theta, length }
    }

    pub fn from_cartesian(v: &Vec2) -> Self {
        Polar {
            theta: (v.y / v.x).atan(),
            length: v.length(),
        }
    }

    pub fn to_cartesian(&self) -> Vec2 {
        Vec2::new(
            self.theta.cos() * self.length,
            self.theta.sin() * self.length,
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct Cylindrical {
    pub theta: f32, // in radians
    pub length: f32,
    pub z: f32,
}

impl Cylindrical {
    pub fn new(theta: f32, length: f32, z: f32) -> Self {
        Cylindrical { theta, length, z }
    }

    pub fn from_cartesian(v: &Vec3) -> Self {
        Cylindrical {
            theta: (v.y / v.x).atan(),
            length: Vec2::new(v.x, v.y).length(),
            z: v.z,
        }
    }

    pub fn to_cartesian(&self) -> Vec3 {
        Vec3::new(
            self.theta.cos() * self.length,
            self.theta.sin() * self.length,
            self.z,
        )
    }
}

// use [ISO 31-11](https://en.wikipedia.org/wiki/ISO_31-11) specification
pub struct Spherical {
    pub length: f32,
    pub theta: f32, // in radians, arccos(z / len)
    pub phi: f32,   // in radians, arctan(y / x)
}

impl Spherical {
    pub fn new(theta: f32, phi: f32, length: f32) -> Self {
        Spherical { length, theta, phi }
    }

    pub fn from_cartesian(v: &Vec3) -> Self {
        let len = v.length();
        Spherical {
            length: len,
            theta: (v.y / v.x).atan(),
            phi: (v.z / len).acos(),
        }
    }

    pub fn to_cartesian(&self) -> Vec3 {
        let xoy_len = self.phi.tan() * self.length;
        Vec3::new(
            xoy_len * self.theta.cos(),
            xoy_len * self.theta.sin(),
            self.length * self.phi.cos(),
        )
    }
}

// Quaternion
struct Quaternion {
    pub s: f32,
    pub v: Vec3,
}

impl Mul<f32> for Quaternion {
    type Output = Self;

    fn mul(self, rhs: f32) -> Self::Output {
        Self {
            s: rhs * self.s,
            v: rhs * self.v,
        }
    }
}

impl Div<f32> for Quaternion {
    type Output = Self;

    fn div(self, rhs: f32) -> Self::Output {
        self * (1.0 / rhs)
    }
}

impl Mul<Quaternion> for f32 {
    type Output = Quaternion;

    fn mul(self, rhs: Quaternion) -> Self::Output {
        rhs * self
    }
}

impl Add for Quaternion {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self {
            s: self.s + rhs.s,
            v: self.v + rhs.v,
        }
    }
}

impl Sub for Quaternion {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        self + (-rhs)
    }
}

impl Neg for Quaternion {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self {
            s: -self.s,
            v: -self.v,
        }
    }
}

impl Quaternion {
    pub fn length_square(&self) -> f32 {
        self.s * self.s + self.v.length_square()
    }

    pub fn length(&self) -> f32 {
        self.length_square().sqrt()
    }

    pub fn conjugate(&self) -> Quaternion {
        Quaternion {
            s: self.s,
            v: -self.v,
        }
    }

    // Hamilton product
    pub fn mul(&self, rhs: &Quaternion) -> Quaternion {
        Quaternion {
            s: self.s * rhs.s - self.v.dot(&rhs.v),
            v: self.s * rhs.v + self.v * rhs.s + self.v.cross(&rhs.v),
        }
    }

    pub fn inverse(&self) -> Quaternion {
        self.conjugate() / self.length()
    }
}

// basic geomentry primitives
pub struct Line3D {
    pub point: Vec3,
    pub dir: Vec3,
}

impl Line3D {
    pub fn new(p: &Vec3, dir: &Vec3) -> Self {
        Line3D {
            point: *p,
            dir: *dir,
        }
    }

    pub fn intersect_plane(&self, plane: &Plane) -> Option<f32> {
        if plane.normal.dot(&self.dir).abs() >= f32::EPSILON {
            let a = plane.point.dot(&plane.normal);
            let b = self.point.dot(&plane.normal);
            let t = (a - b) / (self.dir.x + self.dir.y + self.dir.z);
            return Some(t);
        }
        None
    }
}

pub struct Segment {
    pub a: Vec3,
    pub b: Vec3,
    pub dir: Vec3,
}

impl Segment {
    pub fn is_in_plane(&self, plane: &Plane) -> bool {
        (self.a - plane.point).dot(&plane.normal).abs() <= f32::EPSILON
            && (self.b - plane.point).dot(&plane.normal).abs() <= f32::EPSILON
    }
}

impl Segment {
    pub fn new(a: &Vec3, b: &Vec3, dir: &Vec3) -> Self {
        Segment {
            a: *a,
            b: *b,
            dir: *dir,
        }
    }

    pub fn intersect_plane(&self, plane: &Plane) -> Option<f32> {
        if let Some(t) = Line3D::new(&self.a, &self.dir).intersect_plane(plane) {
            if t >= 0.0 && t <= 1.0 {
                return Some(t);
            }
        }
        None
    }
}

pub struct Plane {
    pub point: Vec3,
    pub normal: Vec3,
}

impl Plane {
    pub fn new(p: &Vec3, n: &Vec3) -> Self {
        Plane {
            point: *p,
            normal: *n,
        }
    }
}

pub type Color = Vec4;

// unittest

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn vector_new() {
        let v2 = Vec2::new(1.0, 2.0);
        assert_eq!(v2.x, 1.0);
        assert_eq!(v2.y, 2.0);

        let v3 = Vec3::new(1.0, 2.0, 3.0);
        assert_eq!(v3.x, 1.0);
        assert_eq!(v3.y, 2.0);
        assert_eq!(v3.z, 3.0);

        let v4 = Vec4::new(1.0, 2.0, 3.0, 0.5);
        assert_eq!(v4.x, 1.0);
        assert_eq!(v4.y, 2.0);
        assert_eq!(v4.z, 3.0);
        assert_eq!(v4.w, 0.5);
    }

    #[test]
    fn vector_eq() {
        let v1 = Vec2::new(1.0, 2.0);
        let v2 = Vec2::new(2.0, 3.0);
        let v3 = Vec2::new(2.0, 3.0);
        assert_ne!(v1, v2);
        assert_eq!(v2, v3);
    }

    #[test]
    fn vector2_math_algorithm() {
        let v1 = Vec2::new(1.0, 2.0);
        let v2 = Vec2::new(4.0, 6.0);

        assert_eq!(v1 + v2, Vec2::new(5.0, 8.0));
        assert_eq!(v1 - v2, Vec2::new(-3.0, -4.0));
        assert_eq!(v1 * v2, Vec2::new(4.0, 12.0));
        assert_eq!(v1 / v2, Vec2::new(0.25, 2.0 / 6.0));
        assert_eq!(v1 * 3.0, Vec2::new(3.0, 6.0));
        assert_eq!(v1 / 2.0, Vec2::new(0.5, 1.0));
        assert_eq!(3.0 * v1, Vec2::new(3.0, 6.0));
        assert_eq!(v1.length_square(), 5.0);
        assert_eq!(v1.length(), 5.0f32.sqrt());
        assert_eq!(v1.cross(&v2), -2.0);
        assert_eq!(v1.dot(&v2), 16.0);
        assert_eq!(v1.normalize(), v1 / v1.length());
    }

    #[test]
    fn vector3_math_algorithm() {
        let v1 = Vec3::new(1.0, 2.0, 3.0);
        let v2 = Vec3::new(4.0, 6.0, 8.0);

        assert_eq!(v1 + v2, Vec3::new(5.0, 8.0, 11.0));
        assert_eq!(v1 - v2, Vec3::new(-3.0, -4.0, -5.0));
        assert_eq!(v1 * v2, Vec3::new(4.0, 12.0, 24.0));
        assert_eq!(v1 / v2, Vec3::new(0.25, 2.0 / 6.0, 3.0 / 8.0));
        assert_eq!(v1 * 3.0, Vec3::new(3.0, 6.0, 9.0));
        assert_eq!(v1 / 2.0, Vec3::new(0.5, 1.0, 1.5));
        assert_eq!(3.0 * v1, Vec3::new(3.0, 6.0, 9.0));
        assert_eq!(v1.length_square(), 14.0);
        assert_eq!(v1.length(), v1.length_square().sqrt());
        assert_eq!(v1.cross(&v2), Vec3::new(-2.0, 4.0, -2.0));
        assert_eq!(v1.dot(&v2), 40.0);
        assert_eq!(v1.normalize(), v1 / v1.length());
    }

    #[test]
    fn vector4_math_algorithm() {
        let v1 = Vec4::new(1.0, 2.0, 3.0, 2.0);
        let v2 = Vec4::new(4.0, 6.0, 8.0, 3.0);

        assert_eq!(v1 + v2, Vec4::new(5.0, 8.0, 11.0, 5.0));
        assert_eq!(v1 - v2, Vec4::new(-3.0, -4.0, -5.0, -1.0));
        assert_eq!(v1 * v2, Vec4::new(4.0, 12.0, 24.0, 6.0));
        assert_eq!(v1 / v2, Vec4::new(0.25, 2.0 / 6.0, 3.0 / 8.0, 2.0 / 3.0));
        assert_eq!(v1 * 3.0, Vec4::new(3.0, 6.0, 9.0, 6.0));
        assert_eq!(v1 / 2.0, Vec4::new(0.5, 1.0, 1.5, 1.0));
        assert_eq!(3.0 * v1, Vec4::new(3.0, 6.0, 9.0, 6.0));
        assert_eq!(v1.length_square(), 18.0);
        assert_eq!(v1.length(), v1.length_square().sqrt());
        assert_eq!(v1.dot(&v2), 46.0);
        assert_eq!(v1.normalize(), v1 / v1.length());
    }

    #[test]
    fn mat_math_algorithm() {
        let m1 = Mat4::from_row(&[
            1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16.,
        ]);
        let m2 = Mat4::from_row(&[
            2., 1., 4., 3., 8., 7., 1., 9., 0., 6., 5., 2., 8., 9., 4., 3.,
        ]);

        let result = m1 * m2;

        let check_result = Mat4::from_row(&[
            50., 69., 37., 39., 122., 161., 93., 107., 194., 253., 149., 175., 266., 345., 205.,
            243.,
        ]);
        assert_eq!(result, check_result);
    }
}
