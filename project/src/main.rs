fn main() {
    let a = [1, 2, 3];
    let b = [3, 4, 5];

    let result = mul(&a, &b);

    println!("{result}");
}

fn mul(a: &[i32; 3], b: &[i32; 3]) -> i32 {
    let mut s = 0;

    for k in 0..2 {
        s += a[k] * b[k]
    }

    s
}