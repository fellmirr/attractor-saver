#!/bin/zsh
# Builds:
#   build/Attractor.saver          the screensaver
#   build/Attractor Viewer.app     a windowed viewer (embeds the saver)
#   build/attractor-render         offline PNG/benchmark harness
# Usage: ./build.sh [install]   — "install" also copies the saver to ~/Library/Screen Savers
set -euo pipefail
cd "${0:A:h}"

if [[ $# -gt 1 || ( $# -eq 1 && "$1" != "install" ) ]]; then
  echo "usage: ./build.sh [install]" >&2
  exit 1
fi

TARGET=arm64-apple-macos15.0
FLAGS=(-O -target $TARGET -swift-version 5)
SHARED=(Sources/Shared/*.swift)

rm -rf build/Attractor.saver "build/Attractor Viewer.app" build/attractor-render
mkdir -p build

# --- Screensaver bundle
SAVER=build/Attractor.saver
mkdir -p $SAVER/Contents/MacOS $SAVER/Contents/Resources
swiftc $FLAGS -module-name AttractorSaver -emit-library \
  -Xlinker -install_name -Xlinker @rpath/AttractorSaver \
  -o $SAVER/Contents/MacOS/AttractorSaver \
  $SHARED Sources/Saver/*.swift \
  -framework ScreenSaver -framework Metal -framework QuartzCore -framework AppKit -framework CoreText
cat > $SAVER/Contents/Info.plist <<'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleDevelopmentRegion</key><string>en</string>
  <key>CFBundleExecutable</key><string>AttractorSaver</string>
  <key>CFBundleIdentifier</key><string>local.attractor.saver</string>
  <key>CFBundleInfoDictionaryVersion</key><string>6.0</string>
  <key>CFBundleName</key><string>Attractor</string>
  <key>CFBundleDisplayName</key><string>Attractor</string>
  <key>CFBundlePackageType</key><string>BNDL</string>
  <key>CFBundleShortVersionString</key><string>1.0</string>
  <key>CFBundleVersion</key><string>1</string>
  <key>NSPrincipalClass</key><string>AttractorSaverView</string>
  <key>LSMinimumSystemVersion</key><string>15.0</string>
</dict>
</plist>
EOF
codesign --force --sign - $SAVER

# --- Viewer app (embeds a copy of the saver)
APP="build/Attractor Viewer.app"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"
swiftc $FLAGS -o "$APP/Contents/MacOS/AttractorViewer" Sources/Host/main.swift -framework ScreenSaver -framework AppKit
cp -R $SAVER "$APP/Contents/Resources/"
cat > "$APP/Contents/Info.plist" <<'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleExecutable</key><string>AttractorViewer</string>
  <key>CFBundleIdentifier</key><string>local.attractor.viewer</string>
  <key>CFBundleName</key><string>Attractor Viewer</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>CFBundleShortVersionString</key><string>1.0</string>
  <key>NSHighResolutionCapable</key><true/>
  <key>LSMinimumSystemVersion</key><string>15.0</string>
</dict>
</plist>
EOF
codesign --force --deep --sign - "$APP"

# --- Offline harness
swiftc $FLAGS -o build/attractor-render $SHARED Sources/Tool/main.swift

echo "Built: $SAVER, $APP, build/attractor-render"

if [[ "${1:-}" == "install" ]]; then
  mkdir -p ~/Library/Screen\ Savers
  rm -rf ~/Library/Screen\ Savers/Attractor.saver
  cp -R $SAVER ~/Library/Screen\ Savers/
  # legacyScreenSaver caches loaded bundles; kill it so the new build is picked up.
  killall legacyScreenSaver 2>/dev/null || true
  killall "legacyScreenSaver (System Settings)" 2>/dev/null || true
  echo "Installed to ~/Library/Screen Savers/Attractor.saver"
fi
