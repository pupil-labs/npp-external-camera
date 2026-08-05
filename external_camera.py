# /// script
# requires-python = ">=3.12"
# dependencies = [
# ]
# ///
import functools
import logging
import math
from enum import Flag, auto
from pathlib import Path
from typing import Any

import numpy as np
import pupil_labs.video as plv
from pupil_labs.neon_player import Plugin, action
from pupil_labs.neon_player.utilities import qimage_from_frame
from pupil_labs.neon_recording import NeonRecording
from PySide6.QtCore import QRectF, Qt
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
from PySide6.QtWidgets import QDockWidget, QLabel, QSizePolicy
from qt_property_widgets.utilities import action_params, property_params

logger = logging.getLogger(__name__)


class ModifyDirection(Flag):
    NONE = 0
    CORNER = auto()
    MOVE = auto()


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
            if str(self.source).lower().endswith((".mov", ".mp4")):
                import av

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
    except Exception as e:
        logger.exception(f"ExternalCameraPlugin: Failed to patch plv.Reader: {e}")


patch_pl_video()


class SimplifiedExternalCamera:
    def __init__(self, mov_path, time_path):
        self.reader = plv.Reader(str(mov_path), stream="video")
        self.timestamps = np.loadtxt(str(time_path), dtype=np.int64)

        diffs = np.diff(self.timestamps)
        self.frame_duration_ns = int(np.median(diffs)) if len(diffs) > 0 else 5_000_000

        self._last_idx = -1
        self._last_frame = None

    def sample(
        self, timestamps_ns: list[int] | np.ndarray, method: str = "backward"
    ) -> list[Any]:
        if len(timestamps_ns) == 0:
            return []

        target_ts = timestamps_ns[0]
        idx = np.searchsorted(self.timestamps, target_ts)
        if idx >= len(self.timestamps):
            idx = len(self.timestamps) - 1
        elif idx > 0 and method == "backward" and self.timestamps[idx] > target_ts:
            idx = max(0, idx - 1)

        if idx != self._last_idx:
            try:
                frame = self.reader[idx]

                class MockSample:
                    def __init__(self, av_frame):
                        try:
                            self.bgr = av_frame.to_ndarray(format="bgr24")
                        except TypeError:
                            try:
                                self.bgr = av_frame.reformat(
                                    format="bgr24"
                                ).to_ndarray()
                            except Exception:
                                rgb_array = np.array(av_frame.to_image())
                                self.bgr = rgb_array[..., ::-1].copy()

                self._last_frame = MockSample(frame)
                self._last_idx = idx
            except Exception as e:
                logger.exception(f"ExternalCamera: Error reading frame {idx}: {e}")
                return []

        return [self._last_frame] if self._last_frame else []

    def __len__(self):
        return len(self.timestamps)


class ExternalCameraPlugin(Plugin):
    label = "External Camera"

    def __init__(self):
        super().__init__()
        self._render_external_camera = True
        self._undocked = False

        self._alpha = 1.0
        self._scale = 1.0
        self._rotation = 0.0
        self._offset_x = 0.02
        self._offset_y = 0.02
        self._anchor_right = True
        self._anchor_bottom = True

        self._mouse_mode = ModifyDirection.NONE
        self._drag_start_pos = None
        self._start_geometry = QRectF()
        self._start_rotation = 0.0
        self._start_scale = 1.0
        self._last_rect = None
        self._last_rendered_time = 0

    def on_recording_loaded(self, recording: NeonRecording) -> None:

        rec_path = Path(self.recording._rec_dir)
        media_files = sorted(rec_path.glob("recording_*.mov")) + sorted(
            rec_path.glob("recording_*.mp4")
        )
        time_files = [f.with_suffix(".time") for f in media_files]

        valid_pairs = [
            (m, t) for m, t in zip(media_files, time_files, strict=False) if t.exists()
        ]

        if not valid_pairs:
            logger.info(
                "ExternalCameraPlugin: No recording_*.mov/mp4/time pairs found in this recording."
            )
            return

        try:
            mov_path, time_path = valid_pairs[0]
            recording.external_camera = SimplifiedExternalCamera(mov_path, time_path)
            self._frame_duration_ns = recording.external_camera.frame_duration_ns
            logger.info(
                f"ExternalCameraPlugin: Loaded stream from {mov_path.name} with median frame duration {self._frame_duration_ns}ns"
            )
        except Exception as e:
            logger.exception(
                f"ExternalCameraPlugin: Failed to initialize simplified stream: {e}"
            )

        vw = self.app.main_window.video_widget
        try:
            vw.mouse_pressed.disconnect(self.on_mouse_pressed)
            vw.mouse_moved.disconnect(self.on_mouse_moved)
            vw.mouse_released.disconnect(self.on_mouse_released)
        except Exception:  # ruff: ignore[blind-except, try-except-pass]
            pass

        vw.mouse_pressed.connect(self.on_mouse_pressed)
        vw.mouse_moved.connect(self.on_mouse_moved)
        vw.mouse_released.connect(self.on_mouse_released)

        self._shortcut_h = QShortcut(QKeySequence("Shift+H"), self.app.main_window)
        self._shortcut_h.activated.connect(self.previous_frame)

        self._shortcut_l = QShortcut(QKeySequence("Shift+L"), self.app.main_window)
        self._shortcut_l.activated.connect(self.next_frame)

        if self._undocked:
            self._setup_dock_widget()

    def on_disabled(self) -> None:
        self._remove_dock_widget()

        try:
            vw = self.app.main_window.video_widget
            vw.mouse_pressed.disconnect(self.on_mouse_pressed)
            vw.mouse_moved.disconnect(self.on_mouse_moved)
            vw.mouse_released.disconnect(self.on_mouse_released)
        except Exception:  # ruff: ignore[blind-except, try-except-pass]
            pass

        if getattr(self, "_shortcut_h", None):
            self._shortcut_h.setEnabled(False)
            self._shortcut_h.deleteLater()
            self._shortcut_h = None

        if getattr(self, "_shortcut_l", None):
            self._shortcut_l.setEnabled(False)
            self._shortcut_l.deleteLater()
            self._shortcut_l = None

    def _setup_dock_widget(self):
        if not hasattr(self, "app") or not getattr(self.app, "main_window", None):
            return
        if hasattr(self, "_dock") and self._dock:
            return

        self._dock = QDockWidget("External Camera", self.app.main_window)
        self._label = QLabel()
        self._label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._label.setSizePolicy(
            QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored
        )
        self._dock.setWidget(self._label)
        self.app.main_window.addDockWidget(
            Qt.DockWidgetArea.RightDockWidgetArea, self._dock
        )

    def _remove_dock_widget(self):
        if hasattr(self, "_dock") and self._dock:
            self.app.main_window.removeDockWidget(self._dock)
            self._dock.deleteLater()
            self._dock = None
            self._label = None

    def render(self, painter: QPainter, time_in_recording: int) -> None:
        self._last_rendered_time = time_in_recording
        if not self._render_external_camera:
            return

        if not hasattr(self.recording, "external_camera"):
            return

        samples = self.external_camera.sample([time_in_recording])
        if not samples:
            return

        sample = samples[0]
        if sample is None or not hasattr(sample, "bgr"):
            return

        img = qimage_from_frame(sample.bgr)
        if not img:
            return

        if self._undocked:
            if hasattr(self, "_label") and self._label:
                pixmap = QPixmap.fromImage(img)
                scaled_pixmap = pixmap.scaled(
                    self._label.size(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
                self._label.setPixmap(scaled_pixmap)
            return

        sw, sh = self.recording.scene.width, self.recording.scene.height
        base_th = sh / 3.0
        th = base_th * self._scale
        tw = th * (img.width() / img.height())

        x = (
            (sw - tw - self._offset_x * sw)
            if self._anchor_right
            else (self._offset_x * sw)
        )
        y = (
            (sh - th - self._offset_y * sh)
            if self._anchor_bottom
            else (self._offset_y * sh)
        )

        rect = QRectF(x, y, tw, th)
        self._last_rect = rect

        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        painter.setOpacity(self._alpha)

        if self._rotation != 0.0:
            cx, cy = rect.center().x(), rect.center().y()
            painter.translate(cx, cy)
            painter.rotate(self._rotation)
            painter.translate(-cx, -cy)

        painter.drawImage(rect, img)

        painter.setBrush(Qt.BrushStyle.NoBrush)
        if self._mouse_mode != ModifyDirection.NONE:
            painter.setPen(QPen(QColor("#6D7BE0"), 4))
            painter.drawRect(rect)
        else:
            painter.setPen(QPen(QColor(255, 255, 255), 2))
            painter.drawRect(rect)

        painter.restore()

    def _get_transform(self) -> QTransform:
        if not self._last_rect:
            return QTransform()
        t = QTransform()
        cx, cy = self._last_rect.center().x(), self._last_rect.center().y()
        t.translate(cx, cy)
        t.rotate(self._rotation)
        t.translate(-cx, -cy)
        return t

    def on_mouse_pressed(self, event):
        if not self._render_external_camera or self._undocked or not self._last_rect:
            return

        vw = self.app.main_window.video_widget
        pos = (
            vw.map_point(event.position())
            if hasattr(event, "position")
            else vw.map_point(event.pos())
        )

        transform = self._get_transform()
        poly = transform.map(self._last_rect)

        if poly.containsPoint(pos, Qt.FillRule.WindingFill):
            self._drag_start_pos = pos
            self._start_geometry = self._last_rect
            self._start_rotation = self._rotation
            self._start_scale = self._scale
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
        self.app.main_window.video_widget.unsetCursor()
        self.changed.emit()

    def on_hover(self, event):
        if not self._render_external_camera or self._undocked or not self._last_rect:
            return

        vw = self.app.main_window.video_widget
        pos = (
            vw.map_point(event.position())
            if hasattr(event, "position")
            else vw.map_point(event.pos())
        )
        margin = max(30, self._last_rect.width() * 0.15)

        transform = self._get_transform()
        poly = transform.map(self._last_rect)

        if poly.containsPoint(pos, Qt.FillRule.WindingFill):
            corners = [
                self._last_rect.topLeft(),
                self._last_rect.topRight(),
                self._last_rect.bottomLeft(),
                self._last_rect.bottomRight(),
            ]

            is_near_corner = False
            for corner in corners:
                corner_screen = transform.map(corner)
                dist = math.hypot(
                    pos.x() - corner_screen.x(), pos.y() - corner_screen.y()
                )
                if dist < margin:
                    is_near_corner = True
                    break

            if is_near_corner:
                self._mouse_mode = ModifyDirection.CORNER
                vw.setCursor(Qt.CursorShape.CrossCursor)
            else:
                self._mouse_mode = ModifyDirection.MOVE
                vw.setCursor(Qt.CursorShape.SizeAllCursor)
        else:
            self._mouse_mode = ModifyDirection.NONE
            vw.unsetCursor()
        vw.update()

    def on_drag(self, event):
        if (
            self._mouse_mode == ModifyDirection.NONE
            or not self._drag_start_pos
            or self._undocked
        ):
            return

        vw = self.app.main_window.video_widget
        pos = (
            vw.map_point(event.position())
            if hasattr(event, "position")
            else vw.map_point(event.pos())
        )

        dx = pos.x() - self._drag_start_pos.x()
        dy = pos.y() - self._drag_start_pos.y()
        sw, sh = self.recording.scene.width, self.recording.scene.height

        if self._mouse_mode == ModifyDirection.MOVE:
            if self._anchor_right:
                self._offset_x -= dx / sw
            else:
                self._offset_x += dx / sw

            if self._anchor_bottom:
                self._offset_y -= dy / sh
            else:
                self._offset_y += dy / sh

            self._drag_start_pos = pos

        elif self._mouse_mode == ModifyDirection.CORNER:
            if event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
                cx = self._start_geometry.center().x()
                cy = self._start_geometry.center().y()
                dx1 = self._drag_start_pos.x() - cx
                dy1 = self._drag_start_pos.y() - cy
                dx2 = pos.x() - cx
                dy2 = pos.y() - cy

                angle1 = math.degrees(math.atan2(dy1, dx1))
                angle2 = math.degrees(math.atan2(dy2, dx2))

                diff = angle2 - angle1
                diff = (diff + 180) % 360 - 180

                new_rot = self._start_rotation + diff
                self._rotation = (new_rot + 180) % 360 - 180
            else:
                t = QTransform()
                cx, cy = (
                    self._start_geometry.center().x(),
                    self._start_geometry.center().y(),
                )
                t.translate(cx, cy)
                t.rotate(self._start_rotation)
                t.translate(-cx, -cy)

                inv_t, invertible = t.inverted()
                if invertible:
                    local_pos = inv_t.map(pos)
                    local_start = inv_t.map(self._drag_start_pos)

                    dist_start = math.hypot(local_start.x() - cx, local_start.y() - cy)
                    dist_current = math.hypot(local_pos.x() - cx, local_pos.y() - cy)

                    if dist_start > 0:
                        scale_factor = dist_current / dist_start
                        self._scale = max(0.1, self._start_scale * scale_factor)

        self.changed.emit()
        vw.update()

    @property
    def render_external_camera(self) -> bool:
        return self._render_external_camera

    @render_external_camera.setter
    def render_external_camera(self, value: bool) -> None:
        self._render_external_camera = value
        self.changed.emit()

    @property
    @property_params(label="Undock Window")
    def undocked(self) -> bool:
        return self._undocked

    @undocked.setter
    def undocked(self, value: bool) -> None:
        self._undocked = value
        if self._undocked:
            self._setup_dock_widget()
        else:
            self._remove_dock_widget()
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

    @action
    @action_params(
        compact=True,
        icon=QIcon.fromTheme("media-seek-backward"),
    )
    def previous_frame(self):
        if not hasattr(self, "external_camera"):
            return
        ts_array = self.external_camera.timestamps
        if len(ts_array) == 0:
            return

        current_time = getattr(self, "_last_rendered_time", 0)

        idx = int(np.searchsorted(ts_array, current_time))
        target_idx = max(0, idx - 1)

        self.app.seek_to(int(ts_array[target_idx]))

    @action
    @action_params(
        compact=True,
        icon=QIcon.fromTheme("media-seek-forward"),
    )
    def next_frame(self):
        if not hasattr(self, "external_camera"):
            return
        ts_array = self.external_camera.timestamps
        if len(ts_array) == 0:
            return

        current_time = getattr(self, "_last_rendered_time", 0)

        idx = int(np.searchsorted(ts_array, current_time, side="right"))
        target_idx = min(len(ts_array) - 1, idx)

        self.app.seek_to(int(ts_array[target_idx]))
