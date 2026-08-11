# /// script
# requires-python = ">=3.12"
# dependencies = [
# ]
# ///
import contextlib
import functools
import logging
import math
from collections.abc import Callable
from enum import Flag, StrEnum, auto
from pathlib import Path
from typing import Any

import av
import numpy as np
import pupil_labs.video as plv
from pupil_labs.neon_player import Plugin, action
from pupil_labs.neon_player.ui import ListPropertyAppenderAction
from pupil_labs.neon_player.utilities import qimage_from_frame
from pupil_labs.neon_recording import NeonRecording, match_ts
from PySide6.QtCore import QObject, QRectF, Qt, Signal
from PySide6.QtGui import (
    QColor,
    QIcon,
    QKeySequence,
    QPainter,
    QPen,
    QPixmap,
    QShortcut,
    QTransform,
)
from PySide6.QtWidgets import QDockWidget, QLabel, QMessageBox, QSizePolicy
from qt_property_widgets.utilities import (
    PersistentPropertiesMixin,
    action_params,
    property_params,
)
from qt_property_widgets.widgets import DynamicComboWidget

logger = logging.getLogger(__name__)
VIDEO_SUFFIXES = (".mov", ".mp4")
OverlayPainter = Callable[[QPainter, float, float, int, str], None]


class DockLabel(QLabel):
    def __init__(self, camera: "ExternalCamera", plugin: "ExternalCameraPlugin"):
        super().__init__()
        self.camera = camera
        self.plugin = plugin

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.plugin and self.camera:
            self.plugin._render_docked(self.camera, self.plugin._last_rendered_time)


class ModifyDirection(Flag):
    NONE = 0
    CORNER = auto()
    MOVE = auto()


class DisplayMode(StrEnum):
    OFF = "Off"
    PIP = "Picture-in-Picture"
    DOCKED = "Docked Window"


def patch_pl_video():
    try:
        if hasattr(plv.Reader, "_is_patched_for_mov"):
            return

        target = plv.Reader._container
        if not hasattr(target, "func"):
            logger.warning(
                "ExternalCameraPlugin: Could not find .func on plv.Reader._container. Monkey-patch skipped."
            )
            return

        old_func = target.func

        @functools.wraps(old_func)
        def patched_func(self):
            if str(self.source).lower().endswith(VIDEO_SUFFIXES):
                container = av.open(str(self.source), format=None)
                for stream in container.streams.video:
                    stream.thread_type = "FRAME"
                return container
            return old_func(self)

        new_cached_prop = functools.cached_property(patched_func)
        new_cached_prop.__set_name__(plv.Reader, "_container")

        plv.Reader._container = new_cached_prop
        plv.Reader._is_patched_for_mov = True

        logger.info(
            "ExternalCameraPlugin: Successfully patched plv.Reader to support .mov and .mp4 files."
        )
    except Exception:
        logger.exception("ExternalCameraPlugin: Failed to patch plv.Reader")


patch_pl_video()


class ExternalCamera(PersistentPropertiesMixin, QObject):
    changed = Signal()

    def __init__(self) -> None:
        super().__init__()
        self._video_path: str = ""
        self._display_mode = DisplayMode.DOCKED
        self._alpha = 1.0
        self._scale = 1.0
        self._rotation = 0.0
        self._offset_x = 0.02
        self._offset_y = 0.02
        self._anchor_right = True
        self._anchor_bottom = True

        self.reader: plv.Reader | None = None
        self.timestamps: np.ndarray = np.zeros(0, dtype=np.int64)
        self.frame_duration_ns = 5_000_000
        self.last_rect: QRectF | None = None

        self._last_idx = -1
        self._last_frame: plv.VideoFrame | None = None
        self._dock: QDockWidget | None = None
        self._label: QLabel | None = None
        self._dock_state: tuple | None = None

        self._adopt_first_detected()

    def _adopt_first_detected(self) -> None:
        plugin = Plugin.get_instance_by_name("ExternalCameraPlugin")
        if plugin is None or plugin.recording is None:
            return

        detected = plugin.detected_videos()
        if detected is None:
            return
        if detected:
            self.video_path = str(detected[0])
        else:
            QMessageBox.warning(
                plugin.app.main_window,
                "No External Cameras",
                "No recording_*.mov/.mp4 with a matching .time file was found.",
            )

    @property
    @property_params(widget=None)
    def name(self) -> str:
        return Path(self._video_path).name if self._video_path else "(no video)"

    @property
    @property_params(
        label="Video",
        widget=DynamicComboWidget,
        options_source="_video_options",
        options_changed_signal="changed",
    )
    def video_path(self) -> str:
        return self._video_path

    @video_path.setter
    def video_path(self, value: str) -> None:
        value = str(value)
        if value == self._video_path:
            return
        self._video_path = value
        self.open()
        self.changed.emit()

    def _video_options(self) -> list[tuple[str, str]]:
        plugin = Plugin.get_instance_by_name("ExternalCameraPlugin")
        videos = (plugin.detected_videos() or []) if plugin else []
        return [(v.name, str(v)) for v in videos]

    @property
    @property_params(label="Display")
    def display_mode(self) -> DisplayMode:
        return self._display_mode

    @display_mode.setter
    def display_mode(self, value: DisplayMode) -> None:
        self._display_mode = DisplayMode(value)
        plugin = Plugin.get_instance_by_name("ExternalCameraPlugin")
        if plugin is not None:
            self.sync_dock(plugin.app.main_window)
        self.changed.emit()

    @property
    @property_params(label="Opacity", min=0.1, max=1.0, step=0.05)
    def opacity(self) -> float:
        return self._alpha

    @opacity.setter
    def opacity(self, value: float):
        self._alpha = float(value)
        self.changed.emit()

    @property
    @property_params(label="Scale", min=0.1, max=5.0, step=0.1)
    def scale(self) -> float:
        return self._scale

    @scale.setter
    def scale(self, value: float):
        self._scale = float(value)
        self.changed.emit()

    @property
    @property_params(label="Rotation (degrees)", min=-180.0, max=180.0, step=1.0)
    def rotation(self) -> float:
        return self._rotation

    @rotation.setter
    def rotation(self, value: float):
        self._rotation = float(value)
        self.changed.emit()

    def open(self) -> None:
        self.reader = None
        self.timestamps = np.zeros(0, dtype=np.int64)
        self._last_idx = -1
        self._last_frame = None

        if not self._video_path:
            return

        mov_path = Path(self._video_path)
        time_path = mov_path.with_suffix(".time")
        if not mov_path.exists() or not time_path.exists():
            return

        try:
            self.reader = plv.Reader(str(mov_path), stream="video")
            self.timestamps = np.loadtxt(str(time_path), dtype=np.int64)
            diffs = np.diff(self.timestamps)
            self.frame_duration_ns = (
                int(np.median(diffs)) if len(diffs) > 0 else 5_000_000
            )
            logger.info(
                f"ExternalCamera: Opened {mov_path.name} (frame duration: {self.frame_duration_ns}ns)"
            )
        except Exception:
            logger.exception(f"ExternalCamera: Failed to open {mov_path.name}")
            self.reader = None

    def close_dock(self, main_window) -> None:
        if self._dock:
            main_window.removeDockWidget(self._dock)
            self._dock.deleteLater()
            self._dock = None
            self._label = None

    def sync_dock(self, main_window) -> None:
        if self._display_mode == DisplayMode.DOCKED:
            if not self._dock:
                self._dock = QDockWidget("External Camera", main_window)
                plugin = Plugin.get_instance_by_name("ExternalCameraPlugin")
                self._label = DockLabel(self, plugin)
                self._label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                self._label.setSizePolicy(
                    QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored
                )
                self._dock.setWidget(self._label)
                main_window.addDockWidget(
                    Qt.DockWidgetArea.RightDockWidgetArea, self._dock
                )
        else:
            self.close_dock(main_window)

    def sample(
        self, timestamps_ns: list[int] | np.ndarray, method: str = "backward"
    ) -> list[Any]:

        indices = match_ts(timestamps_ns, self.timestamps, method)

        frames = []
        for idx in indices:
            idx = int(idx)
            if idx != self._last_idx:
                try:
                    self._last_frame = self.reader[idx]
                    self._last_idx = idx
                except Exception:
                    logger.exception(f"ExternalCamera: Error reading frame {idx}")
                    self._last_frame = None
            frames.append(self._last_frame)

        return frames

    def current_image(self, time_in_recording: int) -> tuple[int, Any] | None:
        if self.reader is None or len(self.timestamps) == 0:
            return None

        idx = np.searchsorted(self.timestamps, time_in_recording)
        if idx >= len(self.timestamps):
            idx = len(self.timestamps) - 1
        elif idx > 0 and self.timestamps[idx] > time_in_recording:
            idx -= 1

        if idx != self._last_idx:
            try:
                self._last_frame = self.reader[idx]
                self._last_idx = idx
            except Exception:
                logger.exception(f"ExternalCamera: Error reading frame {idx}")
                return None

        if self._last_frame is None:
            return None

        img = qimage_from_frame(self._last_frame.bgr)
        if not img:
            return None

        return idx, img

    @property
    @property_params(widget=None)
    def offset_x(self) -> float:
        return self._offset_x

    @offset_x.setter
    def offset_x(self, value: float) -> None:
        self._offset_x = value

    @property
    @property_params(widget=None)
    def offset_y(self) -> float:
        return self._offset_y

    @offset_y.setter
    def offset_y(self, value: float) -> None:
        self._offset_y = value

    @property
    @property_params(widget=None)
    def anchor_right(self) -> bool:
        return self._anchor_right

    @property
    @property_params(widget=None)
    def anchor_bottom(self) -> bool:
        return self._anchor_bottom


class ExternalCameraPlugin(Plugin):
    label = "External Camera"

    def __init__(self) -> None:
        super().__init__()
        self._cameras: list[ExternalCamera] = []
        self._detected_videos: list[Path] | None = None
        self._overlays: list[OverlayPainter] = []

        self._mouse_mode = ModifyDirection.NONE
        self._drag_start_pos = None
        self._start_geometry = QRectF()
        self._start_rotation = 0.0
        self._start_scale = 1.0
        self._drag_camera: ExternalCamera | None = None
        self._last_rendered_time = 0
        self._signals_connected = False

        self._shortcut_h: QShortcut | None = None
        self._shortcut_l: QShortcut | None = None

        self.header_action = ListPropertyAppenderAction("cameras", "+ Add Camera")

    @property
    @property_params(
        primary=True,
        prevent_add=True,
        item_params={"label_field": "name"},
    )
    def cameras(self) -> list[ExternalCamera]:
        return self._cameras

    @cameras.setter
    def cameras(self, value: list[ExternalCamera]) -> None:
        main_window = getattr(self.app, "main_window", None)
        for camera in self._cameras:
            with contextlib.suppress(RuntimeError):
                camera.changed.disconnect(self.on_cameras_changed)
            if main_window is not None and camera not in value:
                camera.close_dock(main_window)

        self._cameras = value

        for camera in self._cameras:
            if camera.reader is None:
                camera.open()
            camera.changed.connect(self.on_cameras_changed)
            camera.changed.emit()

        self.on_cameras_changed()

    def on_cameras_changed(self) -> None:
        main_window = getattr(self.app, "main_window", None)
        if main_window is not None:
            for camera in self._cameras:
                camera.sync_dock(main_window)
        self.changed.emit()

    def add_overlay(self, painter: OverlayPainter) -> None:
        if painter not in self._overlays:
            self._overlays.append(painter)

    def remove_overlay(self, painter: OverlayPainter) -> None:
        if painter in self._overlays:
            self._overlays.remove(painter)

    def detected_videos(self) -> list[Path] | None:
        return self._detected_videos

    def _make_camera(self, path: Path) -> ExternalCamera:
        cam = ExternalCamera()
        cam.video_path = str(path)
        return cam

    def on_recording_loaded(self, recording: NeonRecording) -> None:
        rec_path = Path(self.recording._rec_dir)
        self._detected_videos = sorted(
            path
            for suffix in VIDEO_SUFFIXES
            for path in rec_path.glob(f"recording_*{suffix}")
            if path.with_suffix(".time").exists()
        )

        known = {c.video_path for c in self._cameras}
        added = [
            self._make_camera(path)
            for path in self._detected_videos
            if str(path) not in known
        ]
        if added:
            self.cameras = self._cameras + added
            logger.info(
                f"ExternalCameraPlugin: found {len(added)} new stream(s): "
                f"{', '.join(c.name for c in added)}"
            )
        else:
            self.cameras = self._cameras

        vw = self.app.main_window.video_widget
        if not getattr(self, "_signals_connected", False):
            vw.mouse_pressed.connect(self.on_mouse_pressed)
            vw.mouse_moved.connect(self.on_mouse_moved)
            vw.mouse_released.connect(self.on_mouse_released)
            self._signals_connected = True

        self._shortcut_h = QShortcut(QKeySequence("Shift+H"), self.app.main_window)
        self._shortcut_h.activated.connect(self.previous_frame)

        self._shortcut_l = QShortcut(QKeySequence("Shift+L"), self.app.main_window)
        self._shortcut_l.activated.connect(self.next_frame)

    def on_disabled(self) -> None:
        main_window = getattr(self.app, "main_window", None)
        if main_window is not None:
            for camera in self._cameras:
                camera.close_dock(main_window)

        if getattr(self, "_signals_connected", False):
            try:
                vw = self.app.main_window.video_widget
                vw.mouse_pressed.disconnect(self.on_mouse_pressed)
                vw.mouse_moved.disconnect(self.on_mouse_moved)
                vw.mouse_released.disconnect(self.on_mouse_released)
            except Exception:
                pass  # ruff: ignore
            self._signals_connected = False

        if getattr(self, "_shortcut_h", None):
            self._shortcut_h.setEnabled(False)
            self._shortcut_h.deleteLater()
            self._shortcut_h = None

        if getattr(self, "_shortcut_l", None):
            self._shortcut_l.setEnabled(False)
            self._shortcut_l.deleteLater()
            self._shortcut_l = None

    def render(self, painter: QPainter, time_in_recording: int) -> None:
        self._last_rendered_time = time_in_recording
        for camera in self._cameras:
            if camera.display_mode == DisplayMode.DOCKED:
                self._render_docked(camera, time_in_recording)
            elif camera.display_mode == DisplayMode.PIP:
                self._render_pip(painter, camera, time_in_recording)

    def _render_docked(self, camera: ExternalCamera, time_in_recording: int) -> None:
        if camera._label is None:
            return

        res = camera.current_image(time_in_recording)
        if not res:
            return
        idx, img = res

        label_size = camera._label.size()
        if label_size.width() <= 0 or label_size.height() <= 0:
            return

        state = (
            idx,
            label_size,
            camera.opacity,
            camera.rotation,
            camera.scale,
        )
        if state == camera._dock_state:
            return
        camera._dock_state = state

        pixmap = QPixmap(label_size)
        pixmap.fill(Qt.GlobalColor.transparent)

        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        painter.setOpacity(camera.opacity)

        img_aspect = img.width() / img.height()
        label_aspect = label_size.width() / label_size.height()
        if img_aspect > label_aspect:
            base_w = label_size.width()
            base_h = base_w / img_aspect
        else:
            base_h = label_size.height()
            base_w = base_h * img_aspect

        cx, cy = label_size.width() / 2, label_size.height() / 2
        painter.translate(cx, cy)
        painter.rotate(camera.rotation)
        painter.scale(camera.scale, camera.scale)
        painter.translate(-cx, -cy)

        rect = QRectF(cx - base_w / 2, cy - base_h / 2, base_w, base_h)
        painter.drawImage(rect, img)

        if self._overlays:
            painter.translate(rect.topLeft())
            for overlay in self._overlays:
                overlay(painter, base_w, base_h, time_in_recording, camera.name)

        painter.end()
        camera._label.setPixmap(pixmap)

    def _render_pip(
        self, painter: QPainter, camera: ExternalCamera, time_in_recording: int
    ) -> None:
        res = camera.current_image(time_in_recording)
        if not res:
            return
        _, img = res

        sw, sh = self.recording.scene.width, self.recording.scene.height
        base_th = sh / 3.0
        th = base_th * camera.scale
        tw = th * (img.width() / img.height())

        x = (
            (sw - tw - camera.offset_x * sw)
            if camera.anchor_right
            else (camera.offset_x * sw)
        )
        y = (
            (sh - th - camera.offset_y * sh)
            if camera.anchor_bottom
            else (camera.offset_y * sh)
        )
        rect = QRectF(x, y, tw, th)
        camera.last_rect = rect

        painter.save()
        painter.setOpacity(camera.opacity)

        if camera.rotation != 0.0:
            cx, cy = rect.center().x(), rect.center().y()
            painter.translate(cx, cy)
            painter.rotate(camera.rotation)
            painter.translate(-cx, -cy)

        painter.drawImage(rect, img)

        if self._mouse_mode != ModifyDirection.NONE and self._drag_camera == camera:
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setPen(QPen(QColor("#6D7BE0"), 4))
            painter.drawRect(rect)
        elif self._drag_camera == camera:
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setPen(QPen(QColor(255, 255, 255), 2))
            painter.drawRect(rect)

        if self._overlays:
            painter.translate(rect.topLeft())
            for overlay in self._overlays:
                overlay(painter, tw, th, time_in_recording, camera.name)

        painter.restore()

    def _event_pos(self, vw, event):
        return (
            vw.map_point(event.position())
            if hasattr(event, "position")
            else vw.map_point(event.pos())
        )

    def _visible_cameras(self) -> list[ExternalCamera]:
        return [
            c
            for c in self._cameras
            if c.display_mode != DisplayMode.OFF and len(c.timestamps) > 0
        ]

    def _hit_test(self, pos) -> ExternalCamera | None:
        for camera in reversed(self._visible_cameras()):
            if camera.display_mode == DisplayMode.PIP and camera.last_rect:
                t = QTransform()
                cx, cy = camera.last_rect.center().x(), camera.last_rect.center().y()
                t.translate(cx, cy)
                t.rotate(camera.rotation)
                t.translate(-cx, -cy)
                poly = t.map(camera.last_rect)
                if poly.containsPoint(pos, Qt.FillRule.WindingFill):
                    return camera
        return None

    def on_mouse_pressed(self, event):
        pos = self._event_pos(self.app.main_window.video_widget, event)
        camera = self._hit_test(pos)
        if camera:
            self._drag_camera = camera
            self._drag_start_pos = pos
            self._start_geometry = camera.last_rect
            self._start_rotation = camera.rotation
            self._start_scale = camera.scale
            self.on_hover(event)
            event.accept()

    def on_mouse_moved(self, event):
        if event.buttons() == Qt.MouseButton.LeftButton:
            self.on_drag(event)
        else:
            self.on_hover(event)

    def on_mouse_released(self, event):
        self._mouse_mode = ModifyDirection.NONE
        self._drag_start_pos = None
        self._drag_camera = None
        self.app.main_window.video_widget.unsetCursor()
        self.changed.emit()

    def on_hover(self, event):
        vw = self.app.main_window.video_widget
        pos = self._event_pos(vw, event)

        camera = self._drag_camera or self._hit_test(pos)
        if not camera or not camera.last_rect:
            self._mouse_mode = ModifyDirection.NONE
            vw.unsetCursor()
            vw.update()
            return

        margin = max(30, camera.last_rect.width() * 0.15)
        t = QTransform()
        cx, cy = camera.last_rect.center().x(), camera.last_rect.center().y()
        t.translate(cx, cy)
        t.rotate(camera.rotation)
        t.translate(-cx, -cy)

        near_corner = False
        for corner in (
            camera.last_rect.topLeft(),
            camera.last_rect.topRight(),
            camera.last_rect.bottomLeft(),
            camera.last_rect.bottomRight(),
        ):
            screen_c = t.map(corner)
            if math.hypot(pos.x() - screen_c.x(), pos.y() - screen_c.y()) < margin:
                near_corner = True
                break

        if near_corner:
            self._mouse_mode = ModifyDirection.CORNER
            vw.setCursor(Qt.CursorShape.CrossCursor)
        else:
            self._mouse_mode = ModifyDirection.MOVE
            vw.setCursor(Qt.CursorShape.SizeAllCursor)

        if self._drag_camera is None:
            self._drag_camera = camera
        vw.update()

    def on_drag(self, event):
        camera = self._drag_camera
        if (
            camera is None
            or self._mouse_mode == ModifyDirection.NONE
            or not self._drag_start_pos
        ):
            return

        vw = self.app.main_window.video_widget
        pos = self._event_pos(vw, event)

        sw, sh = self.recording.scene.width, self.recording.scene.height

        if self._mouse_mode == ModifyDirection.MOVE:
            dx = pos.x() - self._drag_start_pos.x()
            dy = pos.y() - self._drag_start_pos.y()

            camera.offset_x += -dx / sw if camera.anchor_right else dx / sw
            camera.offset_y += -dy / sh if camera.anchor_bottom else dy / sh

            self._drag_start_pos = pos

        elif self._mouse_mode == ModifyDirection.CORNER:
            center = self._start_geometry.center()
            cx, cy = center.x(), center.y()

            if event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
                angle1 = math.degrees(
                    math.atan2(
                        self._drag_start_pos.y() - cy, self._drag_start_pos.x() - cx
                    )
                )
                angle2 = math.degrees(math.atan2(pos.y() - cy, pos.x() - cx))
                diff = (angle2 - angle1 + 180) % 360 - 180
                camera.rotation = (self._start_rotation + diff + 180) % 360 - 180
            else:
                transform = QTransform()
                transform.translate(cx, cy)
                transform.rotate(self._start_rotation)
                transform.translate(-cx, -cy)

                inverted, invertible = transform.inverted()
                if invertible:
                    local_pos = inverted.map(pos)
                    local_start = inverted.map(self._drag_start_pos)

                    dist_start = math.hypot(local_start.x() - cx, local_start.y() - cy)
                    dist_current = math.hypot(local_pos.x() - cx, local_pos.y() - cy)

                    if dist_start > 0:
                        camera.scale = max(
                            0.1, self._start_scale * dist_current / dist_start
                        )

        self.changed.emit()
        vw.update()

    def _step_frame(self, forward: bool) -> None:
        visible = self._visible_cameras()
        if not visible:
            return

        ts_array = visible[0].timestamps
        if len(ts_array) == 0:
            return

        if forward:
            idx = int(np.searchsorted(ts_array, self._last_rendered_time, side="right"))
            target_idx = min(len(ts_array) - 1, idx)
        else:
            idx = int(np.searchsorted(ts_array, self._last_rendered_time))
            target_idx = max(0, idx - 1)

        self.app.seek_to(int(ts_array[target_idx]))

    @action
    @action_params(compact=True, icon=QIcon.fromTheme("media-seek-backward"))
    def previous_frame(self):
        self._step_frame(forward=False)

    @action
    @action_params(compact=True, icon=QIcon.fromTheme("media-seek-forward"))
    def next_frame(self):
        self._step_frame(forward=True)
