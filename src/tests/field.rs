use ark_std::{end_timer, start_timer};
use ff::Field;
use rand::{RngCore, SeedableRng};
use rand_xorshift::XorShiftRng;

pub fn random_field_tests<F: Field>(type_name: String) {
    let mut rng = XorShiftRng::from_seed([
        0x59, 0x62, 0xbe, 0x5d, 0x76, 0x3d, 0x31, 0x8d, 0x17, 0xdb, 0x37, 0x32, 0x54, 0x06, 0xbc,
        0xe5,
    ]);

    random_multiplication_tests::<F, _>(&mut rng, type_name.clone());
    random_addition_tests::<F, _>(&mut rng, type_name.clone());
    random_subtraction_tests::<F, _>(&mut rng, type_name.clone());
    random_negation_tests::<F, _>(&mut rng, type_name.clone());
    random_doubling_tests::<F, _>(&mut rng, type_name.clone());
    random_squaring_tests::<F, _>(&mut rng, type_name.clone());
    random_inversion_tests::<F, _>(&mut rng, type_name.clone());
    random_expansion_tests::<F, _>(&mut rng, type_name);

    assert_eq!(F::zero().is_zero().unwrap_u8(), 1);
    {
        let mut z = F::zero();
        z = z.neg();
        assert_eq!(z.is_zero().unwrap_u8(), 1);
    }

    assert!(bool::from(F::zero().invert().is_none()));

    // Multiplication by zero
    {
        let mut a = F::random(&mut rng);
        a.mul_assign(&F::zero());
        assert_eq!(a.is_zero().unwrap_u8(), 1);
    }

    // Addition by zero
    {
        let mut a = F::random(&mut rng);
        let copy = a;
        a.add_assign(&F::zero());
        assert_eq!(a, copy);
    }
}

fn random_multiplication_tests<F: Field, R: RngCore>(mut rng: R, type_name: String) {
    let message = format!("multiplication {}", type_name);
    let start = start_timer!(|| message);
    for _ in 0..1000000 {
        let a = F::random(&mut rng);
        let b = F::random(&mut rng);
        let c = F::random(&mut rng);

        let mut t0 = a; // (a * b) * c
        t0.mul_assign(&b);
        t0.mul_assign(&c);

        let mut t1 = a; // (a * c) * b
        t1.mul_assign(&c);
        t1.mul_assign(&b);

        let mut t2 = b; // (b * c) * a
        t2.mul_assign(&c);
        t2.mul_assign(&a);

        assert_eq!(t0, t1);
        assert_eq!(t1, t2);
    }
    end_timer!(start);
}

fn random_addition_tests<F: Field, R: RngCore>(mut rng: R, type_name: String) {
    let message = format!("addition {}", type_name);
    let start = start_timer!(|| message);
    for _ in 0..1000000 {
        let a = F::random(&mut rng);
        let b = F::random(&mut rng);
        let c = F::random(&mut rng);

        let mut t0 = a; // (a + b) + c
        t0.add_assign(&b);
        t0.add_assign(&c);

        let mut t1 = a; // (a + c) + b
        t1.add_assign(&c);
        t1.add_assign(&b);

        let mut t2 = b; // (b + c) + a
        t2.add_assign(&c);
        t2.add_assign(&a);

        assert_eq!(t0, t1);
        assert_eq!(t1, t2);
    }
    end_timer!(start);
}

fn random_subtraction_tests<F: Field, R: RngCore>(mut rng: R, type_name: String) {
    let message = format!("subtraction {}", type_name);
    let start = start_timer!(|| message);
    for _ in 0..1000000 {
        let a = F::random(&mut rng);
        let b = F::random(&mut rng);

        let mut t0 = a; // (a - b)
        t0.sub_assign(&b);

        let mut t1 = b; // (b - a)
        t1.sub_assign(&a);

        let mut t2 = t0; // (a - b) + (b - a) = 0
        t2.add_assign(&t1);

        assert_eq!(t2.is_zero().unwrap_u8(), 1);
    }
    end_timer!(start);
}

fn random_negation_tests<F: Field, R: RngCore>(mut rng: R, type_name: String) {
    let message = format!("negation {}", type_name);
    let start = start_timer!(|| message);
    for _ in 0..1000000 {
        let a = F::random(&mut rng);
        let mut b = a;
        b = b.neg();
        b.add_assign(&a);

        assert_eq!(b.is_zero().unwrap_u8(), 1);
    }
    end_timer!(start);
}

fn random_doubling_tests<F: Field, R: RngCore>(mut rng: R, type_name: String) {
    let message = format!("doubling {}", type_name);
    let start = start_timer!(|| message);
    for _ in 0..1000000 {
        let mut a = F::random(&mut rng);
        let mut b = a;
        a.add_assign(&b);
        b = b.double();

        assert_eq!(a, b);
    }
    end_timer!(start);
}

fn random_squaring_tests<F: Field, R: RngCore>(mut rng: R, type_name: String) {
    let message = format!("squaring {}", type_name);
    let start = start_timer!(|| message);
    for _ in 0..1000000 {
        let mut a = F::random(&mut rng);
        let mut b = a;
        a.mul_assign(&b);
        b = b.square();

        assert_eq!(a, b);
    }
    end_timer!(start);
}

fn random_inversion_tests<F: Field, R: RngCore>(mut rng: R, type_name: String) {
    assert!(bool::from(F::zero().invert().is_none()));

    let message = format!("inversion {}", type_name);
    let start = start_timer!(|| message);
    for _ in 0..1000000 {
        let mut a = F::random(&mut rng);
        let b = a.invert().unwrap(); // probablistically nonzero
        a.mul_assign(&b);

        assert_eq!(a, F::one());
    }
    end_timer!(start);
}

fn random_expansion_tests<F: Field, R: RngCore>(mut rng: R, type_name: String) {
    let message = format!("expansion {}", type_name);
    let start = start_timer!(|| message);
    for _ in 0..1000000 {
        // Compare (a + b)(c + d) and (a*c + b*c + a*d + b*d)

        let a = F::random(&mut rng);
        let b = F::random(&mut rng);
        let c = F::random(&mut rng);
        let d = F::random(&mut rng);

        let mut t0 = a;
        t0.add_assign(&b);
        let mut t1 = c;
        t1.add_assign(&d);
        t0.mul_assign(&t1);

        let mut t2 = a;
        t2.mul_assign(&c);
        let mut t3 = b;
        t3.mul_assign(&c);
        let mut t4 = a;
        t4.mul_assign(&d);
        let mut t5 = b;
        t5.mul_assign(&d);

        t2.add_assign(&t3);
        t2.add_assign(&t4);
        t2.add_assign(&t5);

        assert_eq!(t0, t2);
    }
    end_timer!(start);
}

use crate::arithmetic::BaseExt;
use crate::arithmetic::MillerLoopResult;
use crate::bn256;
use crate::bn256::Fq;
use crate::bn256::Fq12;
use crate::group::Curve;
use crate::group::Group;
use ark_std::One;
use num_bigint::BigUint;
use num_traits::Num;
use num_traits::ToPrimitive;
use rand_core::OsRng;
use std::str::FromStr;

use std::ops::Mul;
use std::ops::Neg;
#[test]
fn test_checkpairing_with_c_wi() {
    // exp = 6x + 2 + p - p^2 = lambda - p^3
    let hex_str = Fq::MODULUS;
    let hex_str = hex_str
        .strip_prefix("0x")
        .or_else(|| hex_str.strip_prefix("0X"))
        .unwrap_or(hex_str);
    let p_pow3 = &BigUint::from_str_radix(hex_str, 16).unwrap().pow(3_u32);

    //0x1baaa710b0759ad331ec15183177faf68148fd2e5e487f1c2421c372dee2ddcdd45cf150c7e2d75ab87216b02105ec9bf0519bc6772f06e788e401a57040c54eb9b42c6f8f8e030b136a4fdd951c142faf174e7e839ac9157f83d3135ae0c55
    let lambda = BigUint::from_str(
        "10486551571378427818905133077457505975146652579011797175399169355881771981095211883813744499745558409789005132135496770941292989421431235276221147148858384772096778432243207188878598198850276842458913349817007302752534892127325269"
    ).unwrap();

    let (exp, sign) = if lambda > *p_pow3 {
        (lambda - p_pow3, true)
    } else {
        (p_pow3 - lambda, false)
    };

    // prove e(P1, Q1) = e(P2, Q2)
    // namely e(-P1, Q1) * e(P2, Q2) = 1
    let P1 = bn256::G1::random(&mut OsRng);
    let Q2 = bn256::G2::random(&mut OsRng);
    let factor = bn256::Fr::from_raw([3_u64, 0, 0, 0]);
    let P2 = P1.mul(&factor).to_affine();
    let Q1 = Q2.mul(&factor).to_affine();
    let Q1_prepared = bn256::G2Prepared::from(Q1);
    let Q2_prepared = bn256::G2Prepared::from(Q2.to_affine());

    // f^{lambda - p^3} * wi = c^lambda
    // equivalently (f * c_inv)^{lambda - p^3} * wi = c_inv^{-p^3} = c^{p^3}
    assert_eq!(
        Fq12::one(),
        bn256::multi_miller_loop(&[(&P1.neg().to_affine(), &Q1_prepared), (&P2, &Q2_prepared)])
            .final_exponentiation()
            .0,
    );

    let f =
        bn256::multi_miller_loop(&[(&P1.neg().to_affine(), &Q1_prepared), (&P2, &Q2_prepared)]).0;
    println!("Bn254::multi_miller_loop done!");
    let (c, wi) = compute_c_wi(f);
    let c_inv = c.invert().unwrap();
    let hint = if sign {
        f * wi * (c_inv.pow_vartime(exp.to_u64_digits()))
    } else {
        f * wi * (c_inv.pow_vartime(exp.to_u64_digits()).invert().unwrap())
    };

    //6x+2
    // let six_x_2 = BigUint::from_str("29793968203157093288").unwrap();
    // println!("c_lamada-p3={:?}", c_inv.pow_vartime(six_x_2.to_u64_digits()));
    // calc_naf_v(c);
    // calc_naf_f(c);
    assert_eq!(hint, c.pow_vartime(p_pow3.to_u64_digits()));

    assert_eq!(
        Fq12::one(),
        bn256::multi_miller_loop_c_wi(
            c,
            wi,
            &[(&P1.neg().to_affine(), &Q1_prepared), (&P2, &Q2_prepared)]
        )
        .0,
    );
    println!("Accumulated f_c_wi done!");
}

fn tonelli_shanks_cubic(a: Fq12, c: Fq12, s: u32, t: BigUint, k: BigUint) -> Fq12 {
    let mut r = a.pow_vartime(t.to_u64_digits());
    let e = 3_u32.pow(s - 1);
    let exp = 3_u32.pow(s) * &t;

    // compute cubic root of (a^t)^-1, say h
    let (mut h, cc, mut c) = (Fq12::one(), c.pow_vartime([e as u64]), c.invert().unwrap());
    for i in 1..(s as i32) {
        let delta = (s as i32) - i - 1;
        let d = if delta < 0 {
            r.pow_vartime((&exp / 3_u32.pow((-delta) as u32)).to_u64_digits())
        } else {
            r.pow_vartime([3_u32.pow(delta as u32).to_u64().unwrap()])
        };
        if d == cc {
            (h, r) = (h * c, r * c.pow_vartime([3_u64]));
        } else if d == cc.pow_vartime([2_u64]) {
            (h, r) = (
                h * c.pow_vartime([2_u64]),
                r * c.pow_vartime([3_u64]).pow_vartime([2_u64]),
            );
        }
        c = c.pow_vartime([3_u64])
    }

    // recover cubic root of a
    r = a.pow_vartime(k.to_u64_digits()) * h;
    if t == 3_u32 * k + 1_u32 {
        r = r.invert().unwrap();
    }

    assert_eq!(r.pow_vartime([3_u64]), a);
    r
}

// refer from Algorithm 5 of "On Proving Pairings"(https://eprint.iacr.org/2024/640.pdf)
fn compute_c_wi(f: Fq12) -> (Fq12, Fq12) {
    let hex_str = Fq::MODULUS;
    let hex_str = hex_str
        .strip_prefix("0x")
        .or_else(|| hex_str.strip_prefix("0X"))
        .unwrap_or(hex_str);
    let p = BigUint::from_str_radix(hex_str, 16).unwrap();

    let r = BigUint::from_str(
        "21888242871839275222246405745257275088548364400416034343698204186575808495617",
    )
    .unwrap();
    let lambda = BigUint::from_str(
        "10486551571378427818905133077457505975146652579011797175399169355881771981095211883813744499745558409789005132135496770941292989421431235276221147148858384772096778432243207188878598198850276842458913349817007302752534892127325269"
    ).unwrap();
    let s = 3_u32;
    let exp = p.pow(12_u32) - 1_u32;
    let h = &exp / &r;
    let t = &exp / 3_u32.pow(s);
    let k = (&t + 1_u32) / 3_u32;
    let m = &lambda / &r;
    let d = 3_u32;
    let mm = &m / d;

    // let mut prng = ChaCha20Rng::seed_from_u64(0);
    let cofactor_cubic = 3_u32.pow(s - 1) * &t;

    // make f is r-th residue, but it's not cubic residue
    assert_eq!(f.pow_vartime(h.to_u64_digits()), Fq12::one());
    //todo sometimes  f is cubic residue
    // assert_ne!(f.pow_vartime(cofactor_cubic.to_u64_digits()), Fq12::one());

    // sample a proper scalar w which is cubic non-residue
    let w = {
        let (mut w, mut z) = (Fq12::one(), Fq12::one());
        while w == Fq12::one() {
            // choose z which is 3-th non-residue
            let mut legendre = Fq12::one();
            while legendre == Fq12::one() {
                z = Fq12::random(&mut OsRng);
                legendre = z.pow_vartime(cofactor_cubic.to_u64_digits());
            }
            // obtain w which is t-th power of z
            w = z.pow_vartime(t.to_u64_digits());
        }
        w
    };
    // make sure 27-th root w, is 3-th non-residue and r-th residue
    assert_ne!(w.pow_vartime(cofactor_cubic.to_u64_digits()), Fq12::one());
    assert_eq!(w.pow_vartime(h.to_u64_digits()), Fq12::one());

    let wi = if f.pow_vartime(cofactor_cubic.to_u64_digits()) == Fq12::one() {
        println!("f is fq12_one------------");
        Fq12::one()
    } else {
        // just two option, w and w^2, since w^3 must be cubic residue, leading f*w^3 must not be cubic residue
        let mut wi = w;
        if (f * wi).pow_vartime(cofactor_cubic.to_u64_digits()) != Fq12::one() {
            assert_eq!(
                (f * w * w).pow_vartime(cofactor_cubic.to_u64_digits()),
                Fq12::one()
            );
            wi = w * w;
        }
        wi
    };

    assert_eq!(wi.pow_vartime(h.to_u64_digits()), Fq12::one());

    assert_eq!(lambda, &d * &mm * &r);
    // f1 is scaled f
    let f1 = f * wi;

    // r-th root of f1, say f2
    let r_inv = r.modinv(&h).unwrap();
    assert_ne!(r_inv, BigUint::one());
    let f2 = f1.pow_vartime(r_inv.to_u64_digits());
    assert_ne!(f2, Fq12::one());

    // m'-th root of f, say f3
    let mm_inv = mm.modinv(&(r * h)).unwrap();
    assert_ne!(mm_inv, BigUint::one());
    let f3 = f2.pow_vartime(mm_inv.to_u64_digits());
    assert_eq!(f3.pow_vartime(cofactor_cubic.to_u64_digits()), Fq12::one());
    assert_ne!(f3, Fq12::one());

    // d-th (cubic) root, say c
    let c = tonelli_shanks_cubic(f3, w, s, t, k);
    assert_ne!(c, Fq12::one());
    assert_eq!(c.pow_vartime(lambda.to_u64_digits()), f * wi);

    (c, wi)
}

use ark_std::Zero;
use std::ops::{Add, Sub};
fn to_naf(x: &BigUint) -> Vec<i8> {
    let mut x = x.clone();
    let mut z = Vec::new();
    let zero = BigUint::zero();
    let one = BigUint::one();
    let two = &one + &one;
    let four = &two + &two;

    while x > zero {
        if &x % &two == zero {
            z.push(0);
        } else {
            let zi = if &x % &four == one { 1 } else { -1 };
            if zi == 1 {
                x = x.sub(&one);
            } else {
                x = x.add(&one);
            }
            z.push(zi);
        }
        x = x >> 1;
    }

    z
}

const SIX_U_PLUS_2_NAF_V: [i8; 65] = [
    0, 0, 0, 1, 0, 1, 0, -1, 0, 0, 1, -1, 0, 0, 1, 0, 0, 1, 1, 0, -1, 0, 0, 1, 0, -1, 0, 0, 0, 0,
    1, 1, 1, 0, 0, -1, 0, 0, 1, 0, 0, 0, 0, 0, -1, 0, 0, 1, 1, 0, 0, -1, 0, 0, 0, 1, 1, 0, -1, 0,
    0, 1, 0, 1, 1,
];
const SIX_U_PLUS_2_NAF_F: [i8; 66] = [
    0, 0, 0, 1, 0, 1, 0, -1, 0, 0, -1, 0, 0, 0, 1, 0, 0, -1, 0, -1, 0, 0, 0, 1, 0, -1, 0, 0, 0, 0,
    -1, 0, 0, 1, 0, -1, 0, 0, 1, 0, 0, 0, 0, 0, -1, 0, 0, -1, 0, 1, 0, -1, 0, 0, 0, -1, 0, -1, 0,
    0, 0, 1, 0, -1, 0, 1,
];

fn calc_naf_v(c: Fq12) {
    let c_inv = c.invert().unwrap();
    let mut f = c_inv;
    for i in (1..SIX_U_PLUS_2_NAF_V.len()).rev() {
        f.square_assign();

        let x = SIX_U_PLUS_2_NAF_V[i - 1];
        match x {
            1 => {
                f.mul_assign(&c_inv)
                // c_clc.mul_assign(&c_inv)
            }
            -1 => {
                f.mul_assign(&c)
                // c_clc.mul_assign(&c)
            }
            _ => {}
        }
    }
    println!("naf_v f={:?}", f);
}

fn calc_naf_f(c: Fq12) {
    let c_inv = c.invert().unwrap();
    let mut f = c_inv;
    for i in (1..SIX_U_PLUS_2_NAF_F.len()).rev() {
        f.square_assign();

        let x = SIX_U_PLUS_2_NAF_F[i - 1];
        match x {
            1 => {
                f.mul_assign(&c_inv)
                // c_clc.mul_assign(&c_inv)
            }
            -1 => {
                f.mul_assign(&c)
                // c_clc.mul_assign(&c)
            }
            _ => {}
        }
    }
    println!("naf_F f={:?}", f);
}
#[test]
fn test_naf() {
    println!(
        "naf: {:?}",
        to_naf(&BigUint::from_str("29793968203157093288").unwrap())
    );
}
