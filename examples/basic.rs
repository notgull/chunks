// MIT/Apache2 License

use chunks::IteratorExt;
use std::iter;

fn main() {
    // sufficiently large iterator
    let mut t1 = 0usize;
    let mut t2 = 1usize;
    let sli = iter::repeat_with(|| {
        let next_term: usize = t1 + t2;
        let res = t2;
        t1 = t2;
        t2 = next_term;
        res
    }).take(75);

    let coll = sli.chunks::<5>().collect::<Vec<_>>();
    assert_eq!(coll.first(), Some(&[1, 1, 2, 3, 5]));
    println!("{:?}", coll);
}
