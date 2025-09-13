# Building StyleSnap for Windows ARM64

## Prerequisites
1. **Windows ARM64 machine** (or Windows with ARM64 emulation)
2. **Node.js 18+** - Download from [nodejs.org](https://nodejs.org/)
3. **Rust** - Install from [rustup.rs](https://rustup.rs/)

## Build Steps

1. **Clone the repository:**
   ```bash
   git clone https://github.com/ranjan2601/StyleSnap.git
   cd StyleSnap/frontend
   ```

2. **Install dependencies:**
   ```bash
   npm install
   ```

3. **Add Windows ARM64 target:**
   ```bash
   rustup target add aarch64-pc-windows-msvc
   ```

4. **Build the app:**
   ```bash
   npm run tauri build -- --target aarch64-pc-windows-msvc
   ```

## Output
The .exe file will be created at:
```
src-tauri/target/aarch64-pc-windows-msvc/release/bundle/msi/StyleSnap_1.0.0_arm64_en-US.msi
```

## Alternative: Use GitHub Actions
Push to GitHub and the automated workflow will build Windows ARM64 version for you automatically.
