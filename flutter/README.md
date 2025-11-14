# ML Scanner Flutter Client

Camera-driven document scanner that runs the exported
`deeplabv3plus_mobilenet_512.onnx` model locally via ONNX Runtime and paints the
predicted mask over the captured frame.

## Prerequisites

- Flutter 3.19+ with the `camera` plugin enabled on your target platforms
- The ONNX model generated from `ml_scanner_model/checkpoints/`
- A physical device or emulator with a working camera

## Preparing the model asset

1. Copy `ml_scanner_model/checkpoints/deeplabv3plus_mobilenet_512.onnx` into
   `ml_scanner/flutter/assets/models/` (already referenced inside
   `pubspec.yaml`).
2. If you regenerate weights, replace the file with the fresh export and keep
   the same name so the asset manifest stays valid.

## Running the app

```bash
cd ml_scanner/flutter
flutter pub get
flutter run -d <device>
```

The app launches into a live camera preview. Press **Capture** to grab the
current frame, run Deeplab pre/post-processing that mirrors
`ml_scanner_model/web_app/app.py`, and render both the colored overlay (stacked
above the preview) and the grayscale mask in the bottom panel. The latest mask
resolution and inference time are displayed for quick debugging.

> **Note:** Camera access prompts appear the first time on Android/iOS. Accept
> the permission to allow `camera` to stream frames to the ONNX runtime.
