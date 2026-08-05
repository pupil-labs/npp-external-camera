# External Camera Plugin

External camera plugin. Loads an external camera video and presents it in sync with the scene.

Requires a `.mov` format video with `.time` file that includes timestamps for each frame. Shows the external view in a Picture-in-Picture (PiP) window that can be resized, rotated, moved around, or changed in transparency. Also adds frame scrubbing capabilities, accessible with shortcuts (`Shift + H` / `Shift + L`) and the posibility to dock it to the main window to see it side-by-side with the scene camera.

> [!TIP]
> We will publish a guide on how to generate the `.time` file while recording. For now, you can use the external_camera_manual_sync.py plugin to manually sync your external camera video and generate the `.time` file.
> For now, you can use the [external_camera_manual_sync.py](https://github.com/pupil-labs/npp-external-camera-manual-sync).


## Installation
For instructions on how to install and manage Neon Player plugins, please refer to the [Neon Player Plugin Documentation](https://docs.pupil-labs.com/neon/neon-player/plugin-api/#adding-a-plugin).

## Usage
1. Ensure you have your external camera video (`.mov` or `.mp4`) and its synced `.time` file ready in the same directory.
2. In the Neon Player sidebar, click the **Load Video** action and select your external video. The `.time` file will be loaded automatically if it matches the video name.
3. The external view will appear as a Picture-in-Picture (PiP) overlay on top of the scene camera. You can resize, rotate, and move the PiP window freely.
4. You can also dock the external view side-by-side with the scene camera using the plugin settings.
5. Use the shortcuts `Shift + H` and `Shift + L` to scrub frames back and forth precisely.
