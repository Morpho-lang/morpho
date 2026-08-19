// Mutual recursion: isEven captures a forward reference and is a closure export

fn isEven(n) {
    if (n == 0) return true
    return isOdd(n - 1)
}

fn isOdd(n) {
    if (n == 0) return false
    return isEven(n - 1)
}
