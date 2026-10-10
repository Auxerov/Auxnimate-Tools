"""
Visual Anim Graph (Auxnimate) v1.6.0
====================================
Annotation and 3D Motion Graph visual animation and timing addon for Blender 5.0.
Supports characters, armatures (Pose Bones), real-time sync, and bidirectional colored annotations.
Runs via the navigation bar button in the Properties editor (Shift+F7) as a standard built-in panel.
"""

bl_info = {
    "name": "Auxnimate - Visual Anim Graph",
    "author": "Auxerov",
    "version": (1, 6, 5),
    "blender": (5, 0, 0),
    "location": "Properties Editor (Shift+F7) Sidebar",
    "description": "Visual animation generator using 3D annotation trajectories and editable motion graphs.",
    "warning": "",
    "doc_url": "",
    "category": "Animation",
}

import bpy
import mathutils
from bpy.app.handlers import persistent
from mathutils import Vector, kdtree, Matrix
from mathutils.geometry import intersect_point_line_segment, interpolate_bezier


_vag_prev_mode = {}
_vag_is_updating = False
_vag_last_state_cache = {}
_vag_known_graphs = {}


def safe_set_active_object(context, obj, select=True, mode=None, bone_name=None):
    """Context-independent safe object selection, activation, and mode switching."""
    if not obj:
        return
    try:
        for o in list(getattr(context, "selected_objects", [])):
            o.select_set(False)
    except Exception:
        pass

    try:
        context.view_layer.objects.active = obj
        if select:
            obj.select_set(True)
    except Exception as e:
        print("[Auxnimate] Object activation error:", e)

    if mode:
        try:
            if context.object and context.object.mode != mode:
                bpy.ops.object.mode_set(mode=mode)
        except Exception:
            pass

    if mode == "POSE" and bone_name and obj.type == "ARMATURE" and obj.pose:
        try:
            if bone_name in obj.pose.bones:
                obj.data.bones.active = obj.data.bones.get(bone_name)
        except Exception:
            pass


def get_stroke_points_3d(stroke):
    """Returns stroke points as a list of Mathutils Vectors."""
    return [p.co.copy() for p in stroke.points]


def calculate_stroke_length(points):
    """Calculates total Euclidean arc length of a point list."""
    if len(points) < 2:
        return 0.0
    total = 0.0
    for i in range(len(points) - 1):
        total += (points[i + 1] - points[i]).length
    return total


def calculate_cumulative_distances(points):
    """Returns cumulative distance from the start for each polyline vertex."""
    cum_dists = [0.0]
    for i in range(len(points) - 1):
        seg_len = (points[i + 1] - points[i]).length
        cum_dists.append(cum_dists[-1] + seg_len)
    return cum_dists


def build_trajectory_kdtree(traj_points):
    """Builds a KD-Tree from trajectory points."""
    kd = kdtree.KDTree(len(traj_points))
    for i, pt in enumerate(traj_points):
        kd.insert(pt, i)
    kd.balance()
    return kd


def project_marker_onto_trajectory(marker_points, traj_points, cum_dists, kd_tree=None, max_distance=5.0):
    """Projects a short marker stroke onto the trajectory using closest point analysis."""
    if len(marker_points) < 1 or len(traj_points) < 2:
        return None

    best_dist = float("inf")
    best_proj = None
    best_s = 0.0

    sample_points = list(marker_points)
    for i in range(len(marker_points) - 1):
        mid = (marker_points[i] + marker_points[i + 1]) * 0.5
        sample_points.append(mid)

    candidate_indices = set()
    if kd_tree is not None:
        for sp in sample_points:
            for _, idx, dist in kd_tree.find_n(sp, 3):
                if dist <= max_distance * 1.5:
                    candidate_indices.add(max(0, idx - 1))
                    candidate_indices.add(idx)
                    candidate_indices.add(min(len(traj_points) - 2, idx))
    else:
        candidate_indices = set(range(len(traj_points) - 1))

    if not candidate_indices:
        candidate_indices = set(range(len(traj_points) - 1))

    for sp in sample_points:
        for i in candidate_indices:
            if i >= len(traj_points) - 1:
                continue
            p1 = traj_points[i]
            p2 = traj_points[i + 1]
            seg_len = (p2 - p1).length
            if seg_len < 1e-7:
                continue

            proj = intersect_point_line_segment(sp, p1, p2)[0]
            dist = (sp - proj).length

            if dist < best_dist:
                best_dist = dist
                best_proj = proj
                seg_factor = max(0.0, min(1.0, (proj - p1).length / seg_len))
                best_s = cum_dists[i] + (seg_factor * seg_len)

    if best_dist <= max_distance and best_proj is not None:
        return {
            "distance_along_path": best_s,
            "projected_point": best_proj,
            "proximity_distance": best_dist,
        }

    return None


def interpolate_position_on_trajectory(traj_points, cum_dists, s):
    """Returns 3D position and tangent vector at arc length distance s along trajectory."""
    total_len = cum_dists[-1]
    s_clamped = max(0.0, min(total_len, s))

    if s_clamped <= 0.0:
        pos = traj_points[0].copy()
        tangent = (traj_points[1] - traj_points[0]).normalized() if len(traj_points) > 1 else Vector((0, 1, 0))
        return pos, tangent

    if s_clamped >= total_len:
        pos = traj_points[-1].copy()
        tangent = (traj_points[-1] - traj_points[-2]).normalized() if len(traj_points) > 1 else Vector((0, 1, 0))
        return pos, tangent

    for i in range(len(cum_dists) - 1):
        if cum_dists[i] <= s_clamped <= cum_dists[i + 1]:
            seg_len = cum_dists[i + 1] - cum_dists[i]
            if seg_len < 1e-7:
                return traj_points[i].copy(), Vector((0, 1, 0))
            t = (s_clamped - cum_dists[i]) / seg_len
            pos = traj_points[i].lerp(traj_points[i + 1], t)
            tangent = (traj_points[i + 1] - traj_points[i]).normalized()
            return pos, tangent

    return traj_points[-1].copy(), Vector((0, 1, 0))


def calculate_tangent_and_perp(bezier_points, i, tick_len=0.4):
    """Calculates tangent and perpendicular vectors for a Bezier spline point."""
    bp = bezier_points[i]
    num_pts = len(bezier_points)

    if num_pts == 1:
        tangent = Vector((1.0, 0.0, 0.0))
    elif i == 0:
        if (bp.handle_right - bp.co).length > 1e-4:
            tangent = (bp.handle_right - bp.co).normalized()
        else:
            tangent = (bezier_points[1].co - bp.co).normalized()
    elif i == num_pts - 1:
        if (bp.co - bp.handle_left).length > 1e-4:
            tangent = (bp.co - bp.handle_left).normalized()
        else:
            tangent = (bp.co - bezier_points[i - 1].co).normalized()
    else:
        tangent = (bezier_points[i + 1].co - bezier_points[i - 1].co).normalized()

    if tangent.length < 1e-4:
        tangent = Vector((1.0, 0.0, 0.0))

    up = Vector((0.0, 0.0, 1.0))
    if abs(tangent.dot(up)) > 0.95:
        up = Vector((0.0, 1.0, 0.0))
    perp = tangent.cross(up).normalized()

    p_start = bp.co - (perp * tick_len)
    p_end = bp.co + (perp * tick_len)
    return p_start, p_end, tangent


def evaluate_bezier_spline(spline, samples_per_segment=8):
    """Samples Bezier spline into a smooth list of 3D points per segment."""
    bps = spline.bezier_points
    if len(bps) < 2:
        return [bp.co.copy() for bp in bps]

    evaluated = []
    for i in range(len(bps) - 1):
        bp1 = bps[i]
        bp2 = bps[i + 1]
        pts = interpolate_bezier(bp1.co, bp1.handle_right, bp2.handle_left, bp2.co, samples_per_segment)
        if evaluated:
            evaluated.extend(pts[1:])
        else:
            evaluated.extend(pts)
    return evaluated


def set_pose_bone_world_transform(p_bone, arm_obj, world_pos, tangent=None, forward_axis="Y", up_axis="Z", align_rotation=True):
    """
    Converts world target position and tangent orientation into local basis space
    according to bone parent-child hierarchy.
    """
    arm_inv = arm_obj.matrix_world.inverted()
    arm_local_pos = arm_inv @ world_pos

    if p_bone.parent:
        rest_rel = p_bone.bone.parent.matrix_local.inverted() @ p_bone.bone.matrix_local
        parent_transform = p_bone.parent.matrix @ rest_rel
    else:
        parent_transform = p_bone.bone.matrix_local

    basis_translation = parent_transform.inverted() @ arm_local_pos
    p_bone.location = basis_translation

    if align_rotation and tangent is not None and tangent.length > 1e-4:
        try:
            rot_quat_world = tangent.to_track_quat(forward_axis, up_axis)
        except Exception:
            rot_quat_world = tangent.to_track_quat("Y", "Z")

        rot_mat_world = rot_quat_world.to_matrix().to_4x4()
        rot_mat_arm = arm_obj.matrix_world.to_3x3().inverted().to_4x4() @ rot_mat_world
        rot_basis = parent_transform.to_3x3().inverted().to_4x4() @ rot_mat_arm

        if p_bone.rotation_mode == "QUATERNION":
            p_bone.rotation_quaternion = rot_basis.to_quaternion()
        elif p_bone.rotation_mode == "AXIS_ANGLE":
            p_bone.rotation_axis_angle = rot_basis.to_quaternion().to_axis_angle()
        else:
            p_bone.rotation_euler = rot_basis.to_euler(p_bone.rotation_mode)


def get_world_to_bone_loc(p_bone, arm_obj, world_pos):
    """Converts world coordinate point into bone local location space."""
    arm_inv = arm_obj.matrix_world.inverted()
    arm_local = arm_inv @ world_pos
    if p_bone.parent:
        rest_rel = p_bone.bone.parent.matrix_local.inverted() @ p_bone.bone.matrix_local
        parent_transform = p_bone.parent.matrix @ rest_rel
    else:
        parent_transform = p_bone.bone.matrix_local
    return parent_transform.inverted() @ arm_local


def is_layer_hidden(layer):
    """Checks whether an annotation layer is hidden across Blender 5.0 / 4.x and legacy versions."""
    if hasattr(layer, "annotation_hide"):
        return bool(layer.annotation_hide)
    if hasattr(layer, "hide"):
        return bool(layer.hide)
    return False


def gather_all_annotation_strokes(context):
    """
    Gathers all active and visible annotation strokes, excluding auto-generated VAG_* layers to collect user drawings only.
    """
    collected = []
    seen_annotations = []

    if hasattr(context.scene, "annotation") and context.scene.annotation:
        seen_annotations.append(context.scene.annotation)

    for ann in bpy.data.annotations:
        if ann not in seen_annotations:
            seen_annotations.append(ann)

    for ann in seen_annotations:
        if not hasattr(ann, "layers"):
            continue
        for layer in ann.layers:
            if layer.info.startswith("VAG_") or layer.info == "Auxnimate_Graph":
                continue
            if is_layer_hidden(layer):
                continue

            frames_to_check = []
            if hasattr(layer, "active_frame") and layer.active_frame:
                frames_to_check.append(layer.active_frame)
            if hasattr(layer, "frames"):
                for f in layer.frames:
                    if f not in frames_to_check:
                        frames_to_check.append(f)

            for frame in frames_to_check:
                if not hasattr(frame, "strokes"):
                    continue
                for stroke in frame.strokes:
                    if not hasattr(stroke, "points") or len(stroke.points) < 2:
                        continue
                    pts = get_stroke_points_3d(stroke)
                    if len(pts) >= 2:
                        length = calculate_stroke_length(pts)
                        if length > 1e-4:
                            collected.append({
                                "annotation": ann,
                                "layer": layer,
                                "frame": frame,
                                "stroke": stroke,
                                "points": pts,
                                "length": length,
                            })

    return collected


def get_or_create_annotation_datablock(scene):
    """Returns or creates the annotation datablock attached to the scene."""
    ann = scene.annotation
    if not ann:
        ann = bpy.data.annotations.get("Annotations")
        if not ann:
            ann = bpy.data.annotations.new("Annotations")
        scene.annotation = ann
    return ann


def get_or_create_vag_layer(ann, name, color, thickness=3):
    """Returns or creates an AnnotationLayer with the specified name and color."""
    layer = next((l for l in ann.layers if l.info == name), None)
    if not layer:
        layer = ann.layers.new(name)
    layer.color = color
    layer.thickness = thickness
    return layer


def draw_stroke_in_specific_layer(context, ann, target_layer, points_3d):
    """Draws stroke in target layer and safely restores original active drawing layer."""
    if len(points_3d) < 2:
        return None

    prev_idx = ann.layers.active_index
    try:
        idx = list(ann.layers).index(target_layer)
        ann.layers.active_index = idx
        for l in ann.layers:
            l.select = (l == target_layer)

        win = context.window_manager.windows[0]
        area = next((a for a in win.screen.areas if a.type == "VIEW_3D"), None)
        if not area:
            return None
        region = next((r for r in area.regions if r.type == "WINDOW"), None)
        if not region:
            return None

        with context.temp_override(window=win, area=area, region=region):
            stroke_data = []
            for i, p in enumerate(points_3d):
                stroke_data.append({
                    "name": "",
                    "location": p,
                    "mouse": (100 + i * 5, 100 + i * 5),
                    "mouse_event": (100 + i * 5, 100 + i * 5),
                    "x_tilt": 0.0,
                    "y_tilt": 0.0,
                    "time": i * 0.01,
                    "pressure": 1.0,
                    "size": 2.0,
                    "is_start": (i == 0),
                })
            bpy.ops.gpencil.annotate(stroke=stroke_data)

        if not target_layer.frames:
            return None
        last_frame = target_layer.frames[-1]
        if not last_frame.strokes:
            return None
        last_stroke = last_frame.strokes[-1]
        for i, p in enumerate(points_3d):
            last_stroke.points[i].co = p
        return last_stroke
    finally:
        if 0 <= prev_idx < len(ann.layers):
            ann.layers.active_index = prev_idx


def mark_previous_annotations_red(context, all_strokes):
    """
    Turns previous drawn annotations red (using settings.trajectory_color).
    Preserves user drawing color by keeping a dedicated active drawing layer with original brush color.
    """
    if not all_strokes:
        return

    scene = context.scene
    settings = getattr(scene, "aux_anim_graph_settings", None)
    red_color = settings.trajectory_color if settings else (1.0, 0.0, 0.0)

    ann = get_or_create_annotation_datablock(scene)
    if not ann or not hasattr(ann, "layers"):
        return

    source_layers = list({s["layer"] for s in all_strokes if "layer" in s and s["layer"] in list(ann.layers)})
    if not source_layers:
        return

    for layer in source_layers:
        orig_name = layer.info
        orig_color = tuple(layer.color)[:3]

        layer.color = tuple(red_color)[:3]
        if not layer.info.startswith("VAG_"):
            layer.info = f"VAG_Source_{orig_name}"

        clean_name = orig_name if not orig_name.startswith("VAG_") else "Annotations"
        draw_layer = next((l for l in ann.layers if l.info == clean_name and not l.info.startswith("VAG_")), None)
        if not draw_layer:
            draw_layer = ann.layers.new(clean_name)
        draw_layer.color = tuple(orig_color)[:3]
        draw_layer.thickness = 3

        ann.layers.active_index = list(ann.layers).index(draw_layer)


def update_vag_annotations_inplace(context, graph_obj):
    """
    In-place coordinate update for existing VAG_* annotation strokes.
    Does not call any operators, ensuring zero-latency, recursion-free depsgraph updates.
    """
    if not graph_obj or not graph_obj.data or not graph_obj.data.splines:
        return

    ann = context.scene.annotation
    if not ann:
        return

    spline0 = graph_obj.data.splines[0]
    mat = graph_obj.matrix_world
    eval_local = evaluate_bezier_spline(spline0, samples_per_segment=8)
    eval_world = [mat @ p for p in eval_local]

    l_traj = next((l for l in ann.layers if l.info == "VAG_Trajectory"), None)
    if l_traj and l_traj.frames and l_traj.frames[0].strokes:
        st = l_traj.frames[0].strokes[0]
        if len(st.points) == len(eval_world):
            for i, p in enumerate(eval_world):
                st.points[i].co = p

    l_mark = next((l for l in ann.layers if l.info == "VAG_Markers"), None)
    if l_mark and l_mark.frames and l_mark.frames[0].strokes:
        settings = getattr(context.scene, "aux_anim_graph_settings", None)
        tick_sz = settings.graph_tick_size if settings else 0.4
        bps = spline0.bezier_points
        ticks_world = []
        for i in range(len(bps)):
            p1_loc, p2_loc, _ = calculate_tangent_and_perp(bps, i, tick_len=tick_sz)
            ticks_world.append((mat @ p1_loc, mat @ p2_loc))

        if len(l_mark.frames[0].strokes) == len(ticks_world):
            for s_idx, (p1, p2) in enumerate(ticks_world):
                st = l_mark.frames[0].strokes[s_idx]
                if len(st.points) >= 2:
                    st.points[0].co = p1
                    st.points[1].co = p2


def sync_motion_graph_to_annotations(context, graph_obj):
    """
    Transfers 3D Motion Graph curve and timing markers to red 3D annotation layers.
    - Main Trajectory: VAG_Trajectory layer
    - Timing Markers: VAG_Markers layer
    """
    if not graph_obj or graph_obj.type != "CURVE" or not graph_obj.data.splines:
        return False

    settings = context.scene.aux_anim_graph_settings
    traj_color = settings.trajectory_color
    marker_color = settings.marker_color

    spline0 = graph_obj.data.splines[0]
    mat = graph_obj.matrix_world

    eval_local = evaluate_bezier_spline(spline0, samples_per_segment=8)
    eval_world = [mat @ p for p in eval_local]

    ticks_world = []
    for s in list(graph_obj.data.splines)[1:]:
        if len(s.points) >= 2:
            p1 = mat @ Vector((s.points[0].co.x, s.points[0].co.y, s.points[0].co.z))
            p2 = mat @ Vector((s.points[1].co.x, s.points[1].co.y, s.points[1].co.z))
            ticks_world.append((p1, p2))

    ann = get_or_create_annotation_datablock(context.scene)
    l_traj = get_or_create_vag_layer(ann, "VAG_Trajectory", traj_color, thickness=4)
    l_mark = get_or_create_vag_layer(ann, "VAG_Markers", marker_color, thickness=3)

    l_traj.color = traj_color
    l_traj.thickness = 4
    l_mark.color = marker_color
    l_mark.thickness = 3

    f_traj = l_traj.frames[-1] if l_traj.frames else None
    f_mark = l_mark.frames[-1] if l_mark.frames else None

    can_fast_update = (
        f_traj and len(f_traj.strokes) == 1 and len(f_traj.strokes[0].points) == len(eval_world) and
        f_mark and len(f_mark.strokes) == len(ticks_world)
    )

    if can_fast_update:
        for i, p in enumerate(eval_world):
            f_traj.strokes[0].points[i].co = p
        for s_idx, (p1, p2) in enumerate(ticks_world):
            st = f_mark.strokes[s_idx]
            st.points[0].co = p1
            st.points[1].co = p2
        return True

    for f in list(l_traj.frames):
        l_traj.frames.remove(f)
    for f in list(l_mark.frames):
        l_mark.frames.remove(f)

    draw_stroke_in_specific_layer(context, ann, l_traj, eval_world)
    for p1, p2 in ticks_world:
        draw_stroke_in_specific_layer(context, ann, l_mark, [p1, p2])

    return True


def get_action_fcurves_universal(action):
    """Safely returns F-Curves list across Blender 2.8 - 4.3 and 4.4 - 5.0+."""
    if action is None:
        return []

    if hasattr(action, "fcurves") and action.fcurves is not None:
        return list(action.fcurves)

    fcurves_list = []
    if hasattr(action, "layers"):
        for layer in action.layers:
            if hasattr(layer, "strips"):
                for strip in layer.strips:
                    if hasattr(strip, "channelbags"):
                        for bag in strip.channelbags:
                            if hasattr(bag, "fcurves"):
                                fcurves_list.extend(list(bag.fcurves))
    return fcurves_list


def apply_interpolation_to_keyframes(obj, frames, interp_type_str, data_path_filter=None):
    """Applies selected interpolation type to keyframe points."""
    if not obj or not obj.animation_data or not obj.animation_data.action:
        return

    action = obj.animation_data.action
    all_fcurves = get_action_fcurves_universal(action)
    frames_set = set(frames)

    for fc in all_fcurves:
        if data_path_filter and data_path_filter not in fc.data_path:
            continue
        for kp in fc.keyframe_points:
            f_val = int(round(kp.co[0]))
            if f_val in frames_set:
                kp.interpolation = interp_type_str
                if interp_type_str == "BEZIER":
                    kp.handle_left_type = "AUTO_CLAMPED"
                    kp.handle_right_type = "AUTO_CLAMPED"
        fc.update()


def apply_graph_bezier_handles_to_fcurves(obj, frames, bps, mat, loc_data_path="location", is_bone=False, arm_obj=None, p_bone=None):
    """
    Transfers 3D Motion Graph Bezier handles to target Object or Bone location F-Curves so 3D arc curvature reflects directly into animation.
    """
    if not obj or not obj.animation_data or not obj.animation_data.action:
        return
    if len(frames) < 2 or len(bps) < 2:
        return

    n = min(len(frames), len(bps))
    action = obj.animation_data.action
    all_fc = get_action_fcurves_universal(action)
    loc_fcs = {fc.array_index: fc for fc in all_fc if fc.data_path == loc_data_path}
    if not loc_fcs:
        return

    loc_cos = []
    loc_hls = []
    loc_hrs = []
    for bp in bps[:n]:
        w_co = mat @ bp.co
        w_hl = mat @ bp.handle_left
        w_hr = mat @ bp.handle_right
        if is_bone and p_bone and arm_obj:
            loc_cos.append(get_world_to_bone_loc(p_bone, arm_obj, w_co))
            loc_hls.append(get_world_to_bone_loc(p_bone, arm_obj, w_hl))
            loc_hrs.append(get_world_to_bone_loc(p_bone, arm_obj, w_hr))
        elif obj.parent:
            p_inv = obj.parent.matrix_world.inverted()
            loc_cos.append(p_inv @ w_co)
            loc_hls.append(p_inv @ w_hl)
            loc_hrs.append(p_inv @ w_hr)
        else:
            loc_cos.append(w_co)
            loc_hls.append(w_hl)
            loc_hrs.append(w_hr)

    for i in range(n):
        f = frames[i]

        if i > 0:
            delta_f = f - frames[i - 1]
            d_seg = (loc_cos[i] - loc_cos[i - 1]).length
            d_h = (loc_cos[i] - loc_hls[i]).length
            t_offset_l = (d_h / d_seg * delta_f) if d_seg > 1e-4 else (delta_f / 3.0)
            t_offset_l = min(max(t_offset_l, delta_f * 0.05), delta_f * 0.9)
            t_left = f - t_offset_l
        else:
            delta_f = frames[1] - f if n > 1 else 1.0
            t_left = f - delta_f / 3.0

        if i < n - 1:
            delta_f = frames[i + 1] - f
            d_seg = (loc_cos[i + 1] - loc_cos[i]).length
            d_h = (loc_hrs[i] - loc_cos[i]).length
            t_offset_r = (d_h / d_seg * delta_f) if d_seg > 1e-4 else (delta_f / 3.0)
            t_offset_r = min(max(t_offset_r, delta_f * 0.05), delta_f * 0.9)
            t_right = f + t_offset_r
        else:
            delta_f = f - frames[-2] if n > 1 else 1.0
            t_right = f + delta_f / 3.0

        for axis in (0, 1, 2):
            if axis not in loc_fcs:
                continue
            fc = loc_fcs[axis]
            kp = None
            for p in fc.keyframe_points:
                if abs(p.co[0] - f) < 0.5:
                    kp = p
                    break
            if not kp:
                continue

            kp.interpolation = 'BEZIER'
            kp.handle_left_type = 'FREE'
            kp.handle_right_type = 'FREE'
            kp.handle_left = (t_left, loc_hls[i][axis])
            kp.handle_right = (t_right, loc_hrs[i][axis])
            fc.update()


def extract_keyframe_data_from_object(obj):
    """Extracts frame numbers and world coordinates from object location animation."""
    if not obj or not obj.animation_data or not obj.animation_data.action:
        return [], []

    action = obj.animation_data.action
    all_fcurves = get_action_fcurves_universal(action)
    loc_fcurves = [fc for fc in all_fcurves if fc.data_path == "location"]
    if not loc_fcurves:
        return [], []

    unique_frames = sorted({int(round(kp.co[0])) for fc in loc_fcurves for kp in fc.keyframe_points})
    if not unique_frames:
        return [], []

    fc_by_axis = {fc.array_index: fc for fc in loc_fcurves}
    positions = []
    for f in unique_frames:
        x = fc_by_axis[0].evaluate(f) if 0 in fc_by_axis else obj.location.x
        y = fc_by_axis[1].evaluate(f) if 1 in fc_by_axis else obj.location.y
        z = fc_by_axis[2].evaluate(f) if 2 in fc_by_axis else obj.location.z
        local_pos = Vector((x, y, z))
        world_pos = obj.matrix_world @ local_pos if obj.parent else local_pos
        positions.append(world_pos)

    return positions, unique_frames


def extract_keyframe_data_from_bone(scene, arm_obj, bone_name):
    """Extracts frame numbers and world coordinates from Pose Bone location animation."""
    if not arm_obj or not arm_obj.animation_data or not arm_obj.animation_data.action:
        return [], []

    action = arm_obj.animation_data.action
    all_fcurves = get_action_fcurves_universal(action)
    d_path = f'pose.bones["{bone_name}"].location'
    bone_fcurves = [fc for fc in all_fcurves if d_path in fc.data_path]
    if not bone_fcurves:
        return [], []

    unique_frames = sorted({int(round(kp.co[0])) for fc in bone_fcurves for kp in fc.keyframe_points})
    if not unique_frames:
        return [], []

    p_bone = arm_obj.pose.bones.get(bone_name)
    if not p_bone:
        return [], []

    positions = []
    curr_f = scene.frame_current
    try:
        for f in unique_frames:
            scene.frame_set(f)
            bpy.context.view_layer.update()
            world_pos = arm_obj.matrix_world @ p_bone.matrix.translation
            positions.append(world_pos.copy())
    finally:
        scene.frame_set(curr_f)

    return positions, unique_frames


def get_active_target_info(context):
    """Returns animation target info based on current selection and settings."""
    settings = getattr(context.scene, "aux_anim_graph_settings", None)
    target_mode = settings.target_mode if settings else "AUTO"
    active_obj = context.active_object

    if active_obj and active_obj.name.startswith("VAG_Graph_"):
        target_type = active_obj.get("vag_target_type", "OBJECT")
        target_owner = active_obj.get("vag_target_owner", "")
        target_name = active_obj.get("vag_target_name", "")

        if target_type == "BONE" and target_owner:
            arm_obj = bpy.data.objects.get(target_owner)
            if arm_obj and arm_obj.type == "ARMATURE":
                p_bone = arm_obj.pose.bones.get(target_name)
                return {
                    "mode": "BONE",
                    "object": arm_obj,
                    "armature": arm_obj,
                    "pose_bone": p_bone,
                    "name": target_name,
                    "full_label": f"Bone: {arm_obj.name} > {target_name} (Graph Active)",
                }
        elif target_name:
            target_obj = bpy.data.objects.get(target_name)
            if target_obj:
                return {
                    "mode": "OBJECT",
                    "object": target_obj,
                    "armature": None,
                    "pose_bone": None,
                    "name": target_obj.name,
                    "full_label": f"Object: {target_obj.name} (Graph Active)",
                }

    is_armature = bool(active_obj and active_obj.type == "ARMATURE")
    has_active_pbone = bool(context.active_pose_bone)

    if target_mode == "BONE" or (target_mode == "AUTO" and (has_active_pbone or is_armature)):
        arm_obj = None
        if has_active_pbone:
            arm_obj = context.active_pose_bone.id_data
        elif is_armature:
            arm_obj = active_obj

        if arm_obj and arm_obj.type == "ARMATURE":
            p_bone = None
            if settings and settings.target_bone and settings.target_bone in arm_obj.pose.bones:
                p_bone = arm_obj.pose.bones[settings.target_bone]
            elif context.active_pose_bone and context.active_pose_bone.id_data == arm_obj:
                p_bone = context.active_pose_bone
            elif arm_obj.data.bones.active and arm_obj.data.bones.active.name in arm_obj.pose.bones:
                p_bone = arm_obj.pose.bones[arm_obj.data.bones.active.name]
            elif arm_obj.pose.bones:
                p_bone = arm_obj.pose.bones[0]

            if p_bone:
                return {
                    "mode": "BONE",
                    "object": arm_obj,
                    "armature": arm_obj,
                    "pose_bone": p_bone,
                    "name": p_bone.name,
                    "full_label": f"Bone: {arm_obj.name} > {p_bone.name}",
                }
            else:
                return {
                    "mode": "NONE",
                    "object": arm_obj,
                    "armature": arm_obj,
                    "pose_bone": None,
                    "name": "",
                    "full_label": f"Armature: {arm_obj.name} (No Bone Selected!)",
                }

    if active_obj:
        return {
            "mode": "OBJECT",
            "object": active_obj,
            "armature": None,
            "pose_bone": None,
            "name": active_obj.name,
            "full_label": f"Object: {active_obj.name}",
        }

    last_name = context.scene.get("vag_last_target_name", "")
    if last_name:
        candidate = bpy.data.objects.get(last_name)
        if candidate:
            last_type = context.scene.get("vag_last_target_type", "OBJECT")
            if last_type == "BONE" and candidate.type == "ARMATURE":
                b_name = context.scene.get("vag_last_target_bone", "")
                p_bone = candidate.pose.bones.get(b_name) if b_name else (candidate.pose.bones[0] if candidate.pose.bones else None)
                return {
                    "mode": "BONE",
                    "object": candidate,
                    "armature": candidate,
                    "pose_bone": p_bone,
                    "name": p_bone.name if p_bone else "",
                    "full_label": f"Bone: {candidate.name} > {p_bone.name if p_bone else ''} (Cached)",
                }
            else:
                return {
                    "mode": "OBJECT",
                    "object": candidate,
                    "armature": None,
                    "pose_bone": None,
                    "name": candidate.name,
                    "full_label": f"Object: {candidate.name} (Cached)",
                }

    return {
        "mode": "NONE",
        "object": None,
        "armature": None,
        "pose_bone": None,
        "name": "",
        "full_label": "No Target Selected!",
    }


def get_motion_graph_name_for_target(target_info):
    """Returns standard graph object name based on target type."""
    if target_info["mode"] == "BONE":
        return f"VAG_Graph_{target_info['armature'].name}_{target_info['pose_bone'].name}"
    elif target_info["mode"] == "OBJECT" and target_info["object"]:
        return f"VAG_Graph_{target_info['object'].name}"
    return "VAG_Graph_Helper"


def build_motion_graph_curve(context, target_info, positions_world, frame_list, tick_size=0.4):
    """Creates an editable 3D Bezier curve in the Viewport for target Object or Bone."""
    graph_name = get_motion_graph_name_for_target(target_info)

    existing = bpy.data.objects.get(graph_name)
    if existing:
        bpy.data.objects.remove(existing, do_unlink=True)

    curve_data = bpy.data.curves.new(f"Curve_{graph_name}", type="CURVE")
    curve_data.dimensions = "3D"
    curve_data.resolution_u = 12

    spline0 = curve_data.splines.new("BEZIER")
    spline0.bezier_points.add(len(positions_world) - 1)

    for i, (pos, _) in enumerate(zip(positions_world, frame_list)):
        bp = spline0.bezier_points[i]
        bp.co = pos
        bp.handle_left_type = "AUTO"
        bp.handle_right_type = "AUTO"

    for bp in spline0.bezier_points:
        bp.handle_left_type = "ALIGNED"
        bp.handle_right_type = "ALIGNED"

    for i in range(len(spline0.bezier_points)):
        p1, p2, _ = calculate_tangent_and_perp(spline0.bezier_points, i, tick_len=tick_size)
        sp = curve_data.splines.new("POLY")
        sp.points.add(1)
        sp.points[0].co = (p1.x, p1.y, p1.z, 1.0)
        sp.points[1].co = (p2.x, p2.y, p2.z, 1.0)

    graph_obj = bpy.data.objects.new(graph_name, curve_data)
    context.collection.objects.link(graph_obj)
    graph_obj.show_in_front = True

    if target_info["mode"] == "BONE":
        graph_obj["vag_target_type"] = "BONE"
        graph_obj["vag_target_owner"] = target_info["armature"].name
        graph_obj["vag_target_name"] = target_info["pose_bone"].name
        context.scene["vag_last_target_bone"] = target_info["pose_bone"].name
    else:
        graph_obj["vag_target_type"] = "OBJECT"
        graph_obj["vag_target_owner"] = target_info["object"].name
        graph_obj["vag_target_name"] = target_info["object"].name

    graph_obj["vag_frames"] = list(frame_list)

    context.scene["vag_last_target_type"] = graph_obj["vag_target_type"]
    context.scene["vag_last_target_owner"] = graph_obj["vag_target_owner"]
    context.scene["vag_last_target_name"] = graph_obj["vag_target_name"]

    global _vag_known_graphs
    _vag_known_graphs[graph_obj.name] = {
        "type": graph_obj["vag_target_type"],
        "owner": graph_obj["vag_target_owner"],
        "name": graph_obj["vag_target_name"],
    }

    return graph_obj


def rebuild_graph_ticks(graph_obj, tick_size=0.4):
    """Rebuilds perpendicular tick lines along Bezier curve points (Called only in Object Mode)."""
    if not graph_obj or graph_obj.type != "CURVE" or not graph_obj.data.splines:
        return

    curve_data = graph_obj.data
    for s in list(curve_data.splines)[1:]:
        curve_data.splines.remove(s)

    bps = curve_data.splines[0].bezier_points
    for i in range(len(bps)):
        p1, p2, _ = calculate_tangent_and_perp(bps, i, tick_len=tick_size)
        sp = curve_data.splines.new("POLY")
        sp.points.add(1)
        sp.points[0].co = (p1.x, p1.y, p1.z, 1.0)
        sp.points[1].co = (p2.x, p2.y, p2.z, 1.0)


def apply_motion_graph_to_target(context, graph_obj):
    """
    Reads Bezier points from VAG_Graph_* curve and updates target Location and Rotation keyframes.
    Transfers 3D Bezier handles directly to animation curves, updates viewport, and syncs annotations.
    """
    global _vag_is_updating
    if not graph_obj or "vag_target_name" not in graph_obj:
        return False

    target_type = graph_obj.get("vag_target_type", "OBJECT")
    target_owner_name = graph_obj.get("vag_target_owner", "")
    target_name = graph_obj.get("vag_target_name", "")
    frames_raw = graph_obj.get("vag_frames", [])
    frames = list(frames_raw)

    if not graph_obj.data or not graph_obj.data.splines:
        return False

    settings = getattr(context.scene, "aux_anim_graph_settings", None)
    if not settings:
        return False

    spline0 = graph_obj.data.splines[0]
    if spline0.type != "BEZIER":
        return False

    mat = graph_obj.matrix_world
    bps = spline0.bezier_points
    if len(bps) == 0:
        return False

    if len(bps) != len(frames):
        start_f = frames[0] if frames else context.scene.frame_current
        end_f = frames[-1] if len(frames) > 1 else (start_f + (len(bps) - 1) * settings.frame_step)
        new_frames = []
        n_bps = len(bps)
        for idx in range(n_bps):
            if n_bps > 1:
                f_val = round(start_f + (idx / float(n_bps - 1)) * (end_f - start_f))
            else:
                f_val = start_f
            if new_frames and f_val <= new_frames[-1]:
                f_val = new_frames[-1] + 1
            new_frames.append(int(f_val))
        frames = new_frames
        graph_obj["vag_frames"] = list(frames)

    _vag_is_updating = True
    try:
        if target_type == "BONE":
            arm_obj = bpy.data.objects.get(target_owner_name)
            if not arm_obj or arm_obj.type != "ARMATURE":
                return False
            p_bone = arm_obj.pose.bones.get(target_name)
            if not p_bone:
                return False

            for i, bp in enumerate(bps):
                if i >= len(frames):
                    break
                current_f = frames[i]
                world_pos = mat @ bp.co
                _, _, tangent = calculate_tangent_and_perp(bps, i)

                set_pose_bone_world_transform(
                    p_bone,
                    arm_obj,
                    world_pos,
                    tangent=tangent,
                    forward_axis=settings.forward_axis,
                    up_axis=settings.up_axis,
                    align_rotation=settings.align_rotation,
                )

                p_bone.keyframe_insert(data_path="location", frame=current_f)
                if settings.align_rotation:
                    if p_bone.rotation_mode == "QUATERNION":
                        p_bone.keyframe_insert(data_path="rotation_quaternion", frame=current_f)
                    elif p_bone.rotation_mode == "AXIS_ANGLE":
                        p_bone.keyframe_insert(data_path="rotation_axis_angle", frame=current_f)
                    else:
                        p_bone.keyframe_insert(data_path="rotation_euler", frame=current_f)

            if settings.interpolation == "BEZIER":
                apply_graph_bezier_handles_to_fcurves(
                    arm_obj,
                    frames,
                    bps,
                    mat,
                    loc_data_path=f'pose.bones["{target_name}"].location',
                    is_bone=True,
                    arm_obj=arm_obj,
                    p_bone=p_bone,
                )
            else:
                apply_interpolation_to_keyframes(
                    arm_obj,
                    frames,
                    settings.interpolation,
                    data_path_filter=f'pose.bones["{target_name}"].location',
                )

            if settings.align_rotation:
                for rot_path in (
                    f'pose.bones["{target_name}"].rotation_euler',
                    f'pose.bones["{target_name}"].rotation_quaternion',
                    f'pose.bones["{target_name}"].rotation_axis_angle',
                ):
                    apply_interpolation_to_keyframes(
                        arm_obj,
                        frames,
                        settings.interpolation if settings.interpolation != "BEZIER" else "BEZIER",
                        data_path_filter=rot_path,
                    )

        else:
            target = bpy.data.objects.get(target_name)
            if not target:
                return False

            for i, bp in enumerate(bps):
                if i >= len(frames):
                    break
                current_f = frames[i]
                world_pos = mat @ bp.co

                if target.parent:
                    loc = target.parent.matrix_world.inverted() @ world_pos
                else:
                    loc = world_pos

                target.location = loc
                target.keyframe_insert(data_path="location", frame=current_f)

                if settings.align_rotation:
                    _, _, tangent = calculate_tangent_and_perp(bps, i)
                    if tangent.length > 1e-4:
                        track_code = settings.forward_axis
                        up_code = settings.up_axis
                        try:
                            rot_quat = tangent.to_track_quat(track_code, up_code)
                        except Exception:
                            rot_quat = tangent.to_track_quat("Y", "Z")

                        if target.parent:
                            rot_mat = target.parent.matrix_world.to_3x3().inverted() @ rot_quat.to_matrix()
                            rot_quat = rot_mat.to_quaternion()

                        if target.rotation_mode == "QUATERNION":
                            target.rotation_quaternion = rot_quat
                            target.keyframe_insert(data_path="rotation_quaternion", frame=current_f)
                        elif target.rotation_mode == "AXIS_ANGLE":
                            target.rotation_axis_angle = rot_quat.to_axis_angle()
                            target.keyframe_insert(data_path="rotation_axis_angle", frame=current_f)
                        else:
                            target.rotation_euler = rot_quat.to_euler(target.rotation_mode)
                            target.keyframe_insert(data_path="rotation_euler", frame=current_f)

            if settings.interpolation == "BEZIER":
                apply_graph_bezier_handles_to_fcurves(
                    target,
                    frames,
                    bps,
                    mat,
                    loc_data_path="location",
                    is_bone=False,
                )
            else:
                apply_interpolation_to_keyframes(target, frames, settings.interpolation)

            if settings.align_rotation:
                for rot_path in ("rotation_euler", "rotation_quaternion", "rotation_axis_angle"):
                    apply_interpolation_to_keyframes(
                        target,
                        frames,
                        settings.interpolation if settings.interpolation != "BEZIER" else "BEZIER",
                        data_path_filter=rot_path,
                    )

        if graph_obj.mode == "OBJECT":
            rebuild_graph_ticks(graph_obj, tick_size=settings.graph_tick_size)

        if settings.auto_sync_annotation:
            update_vag_annotations_inplace(context, graph_obj)

        context.scene.frame_set(context.scene.frame_current)
        if target_type == "BONE":
            arm_obj.update_tag(refresh={'DATA', 'TIME'})
        else:
            target.update_tag(refresh={'DATA', 'TIME'})

        context.view_layer.update()

        for window in context.window_manager.windows:
            for area in window.screen.areas:
                if area.type == 'VIEW_3D':
                    area.tag_redraw()

    finally:
        _vag_is_updating = False

    return True


@persistent
def update_vag_realtime(scene, depsgraph=None):
    """
    Transmits any point, handle, or transform changes on Motion Graph to target and annotations in real time.
    Automatically restores target selection if user deletes the graph in the 3D Viewport.
    """
    global _vag_is_updating, _vag_prev_mode, _vag_last_state_cache, _vag_known_graphs
    if _vag_is_updating:
        return

    context = bpy.context
    if not context or not hasattr(context, "screen"):
        return
    if context.screen and getattr(context.screen, "is_animation_playing", False):
        return

    deleted_names = [name for name in list(_vag_known_graphs.keys()) if name not in bpy.data.objects]
    for d_name in deleted_names:
        info = _vag_known_graphs.pop(d_name, None)
        _vag_prev_mode.pop(d_name, None)
        _vag_last_state_cache.pop(d_name, None)

        if context.view_layer.objects.active is None and info:
            restore_name = info.get("owner") or info.get("name")
            target_obj = bpy.data.objects.get(restore_name)
            if target_obj:
                try:
                    mode = "POSE" if (info.get("type") == "BONE" and target_obj.type == "ARMATURE") else "OBJECT"
                    safe_set_active_object(context, target_obj, select=True, mode=mode, bone_name=info.get("name"))
                    context.scene.aux_properties_tab_active = True
                    for window in context.window_manager.windows:
                        for area in window.screen.areas:
                            if area.type == 'PROPERTIES':
                                area.tag_redraw()
                except Exception as ex:
                    print(f"[Auxnimate] Target restore notice: {ex}")

    helpers = [o for o in bpy.data.objects if o.name.startswith("VAG_Graph_")]
    for g in helpers:
        if g.name not in _vag_known_graphs:
            _vag_known_graphs[g.name] = {
                "type": g.get("vag_target_type", "OBJECT"),
                "owner": g.get("vag_target_owner", ""),
                "name": g.get("vag_target_name", ""),
            }

    settings = getattr(scene, "aux_anim_graph_settings", None)
    if settings and not settings.auto_sync_graph:
        return

    if not helpers:
        return

    for path_obj in helpers:
        curr_mode = path_obj.mode
        prev_mode = _vag_prev_mode.get(path_obj.name, None)
        exited_edit = (prev_mode == "EDIT" and curr_mode == "OBJECT")
        _vag_prev_mode[path_obj.name] = curr_mode

        if not path_obj.data or not path_obj.data.splines:
            continue

        spline0 = path_obj.data.splines[0]
        if spline0.type != "BEZIER":
            continue

        mat = path_obj.matrix_world
        mat_tuple = tuple(round(v, 4) for row in mat for v in row)
        bps_tuple = tuple(
            (
                round(bp.co.x, 4), round(bp.co.y, 4), round(bp.co.z, 4),
                round(bp.handle_left.x, 4), round(bp.handle_left.y, 4), round(bp.handle_left.z, 4),
                round(bp.handle_right.x, 4), round(bp.handle_right.y, 4), round(bp.handle_right.z, 4),
            )
            for bp in spline0.bezier_points
        )
        current_state = (mat_tuple, bps_tuple)
        last_state = _vag_last_state_cache.get(path_obj.name)

        if current_state == last_state and not exited_edit:
            continue

        _vag_last_state_cache[path_obj.name] = current_state

        try:
            apply_motion_graph_to_target(context, path_obj)
        except Exception as e:
            print(f"[Auxnimate] Real-time update notice: {e}")


class AUX_AnimGraphSettings(bpy.types.PropertyGroup):
    target_mode: bpy.props.EnumProperty(
        name="Target Mode",
        description="Target type for animation generation",
        items=[
            ("AUTO", "Auto (Selection)", "Automatically detects active Object or Pose Bone"),
            ("OBJECT", "Object", "Applies animation only to the active object"),
            ("BONE", "Pose Bone", "Applies animation to the selected bone in the active Armature"),
        ],
        default="AUTO",
    )

    target_bone: bpy.props.StringProperty(
        name="Target Bone",
        description="Name of bone to animate (Leave blank to use active pose bone)",
        default="",
    )

    timing_mode: bpy.props.EnumProperty(
        name="Timing Mode",
        description="Method for calculating keyframe timing",
        items=[
            ("STEP", "By Step", "Constant frame step for each timing tick (2D timing chart)"),
            ("DURATION", "By Duration", "Evenly distributes timing ticks across total duration"),
        ],
        default="STEP",
    )

    interpolation: bpy.props.EnumProperty(
        name="Interpolation",
        description="Transition curve type between keyframes",
        items=[
            ("BEZIER", "Bezier", "Smooth curve transitions (Standard)", "IPO_BEZIER", 0),
            ("LINEAR", "Linear", "Linear / direct movement", "IPO_LINEAR", 1),
            ("CONSTANT", "Constant", "Stepped / pose-to-pose preview", "IPO_CONSTANT", 2),
        ],
        default="BEZIER",
    )

    frame_step: bpy.props.IntProperty(
        name="Frame Step",
        description="Frame interval between consecutive timing ticks (e.g. 2 = every 2 frames)",
        default=2,
        min=1,
        max=1000,
    )

    total_duration: bpy.props.IntProperty(
        name="Total Duration",
        description="Total duration for the entire animation in frames",
        default=60,
        min=2,
        max=100000,
    )

    start_frame: bpy.props.IntProperty(
        name="Start Frame",
        description="Starting frame for the generated animation",
        default=1,
        min=-100000,
        max=100000,
    )

    use_current_frame: bpy.props.BoolProperty(
        name="Use Current Frame",
        description="Use the current timeline frame as starting frame",
        default=False,
    )

    align_rotation: bpy.props.BoolProperty(
        name="Align Rotation to Path",
        description="Automatically align target rotation along the trajectory direction/tangent",
        default=True,
    )

    forward_axis: bpy.props.EnumProperty(
        name="Forward Axis",
        description="Local axis of target pointing along the path",
        items=[
            ("Y", "+Y Axis (Blender Default)", ""),
            ("X", "+X Axis", ""),
            ("Z", "+Z Axis", ""),
            ("-Y", "-Y Axis", ""),
            ("-X", "-X Axis", ""),
            ("-Z", "-Z Axis", ""),
        ],
        default="Y",
    )

    up_axis: bpy.props.EnumProperty(
        name="Up Axis",
        description="Local axis of target pointing upward",
        items=[
            ("Z", "+Z Axis (Default)", ""),
            ("Y", "+Y Axis", ""),
            ("X", "+X Axis", ""),
        ],
        default="Z",
    )

    include_endpoints: bpy.props.BoolProperty(
        name="Include Endpoints",
        description="Add keyframes at the exact start and end of trajectory",
        default=True,
    )

    max_proximity: bpy.props.FloatProperty(
        name="Proximity Threshold",
        description="Strokes farther than this distance from trajectory are ignored",
        default=3.0,
        min=0.01,
        max=100.0,
        unit="LENGTH",
    )

    clear_existing: bpy.props.BoolProperty(
        name="Clear Existing Keyframes",
        description="Clear existing transform keyframes on target before generating",
        default=False,
    )

    update_timeline_range: bpy.props.BoolProperty(
        name="Update Scene Frame Range",
        description="Adjust scene start and end frames to fit generated animation",
        default=True,
    )

    jump_to_last_frame: bpy.props.BoolProperty(
        name="Jump to Last Keyframe",
        description="Set current timeline frame to the last generated keyframe",
        default=True,
    )

    auto_snap_cursor: bpy.props.BoolProperty(
        name="Auto-Snap Cursor to Target",
        description="Automatically snap the 3D Cursor to target location and rotation at the final keyframe",
        default=True,
    )

    auto_use_current_frame: bpy.props.BoolProperty(
        name="Auto-Enable Current Frame",
        description="Automatically enable 'Current Frame' after generating animation so subsequent animations chain smoothly",
        default=True,
    )

    graph_tick_size: bpy.props.FloatProperty(
        name="Tick Size",
        description="Visual length of perpendicular timing tick lines on Motion Graph",
        default=0.4,
        min=0.05,
        max=10.0,
        unit="LENGTH",
    )

    auto_sync_graph: bpy.props.BoolProperty(
        name="Real-Time Motion Graph Sync",
        description="Update target animation in real-time as graph curve and handles are edited",
        default=True,
    )

    auto_sync_annotation: bpy.props.BoolProperty(
        name="Update 3D Annotations",
        description="Update red 3D annotation lines when motion graph is edited",
        default=True,
    )

    trajectory_color: bpy.props.FloatVectorProperty(
        name="Trajectory Color",
        description="Color of trajectory curve annotation (Default: Red)",
        subtype="COLOR",
        default=(1.0, 0.0, 0.0),
        size=3,
        min=0.0,
        max=1.0,
    )

    marker_color: bpy.props.FloatVectorProperty(
        name="Marker Color",
        description="Color of timing tick markers (Default: Red)",
        subtype="COLOR",
        default=(1.0, 0.0, 0.0),
        size=3,
        min=0.0,
        max=1.0,
    )


def snap_cursor_to_target_helper(context, target_info):
    """Snaps the 3D Cursor to the location and rotation of the active target Bone or Object."""
    if not target_info or target_info.get("mode") == "NONE":
        return False
    if target_info.get("mode") == "BONE":
        arm_obj = target_info.get("armature")
        p_bone = target_info.get("pose_bone")
        if not arm_obj or not p_bone:
            return False
        context.view_layer.update()
        w_mat = arm_obj.matrix_world @ p_bone.matrix
        context.scene.cursor.location = w_mat.translation
        context.scene.cursor.rotation_euler = w_mat.to_euler()
        return True
    elif target_info.get("mode") == "OBJECT":
        target_obj = target_info.get("object")
        if not target_obj:
            return False
        context.view_layer.update()
        context.scene.cursor.location = target_obj.matrix_world.translation
        context.scene.cursor.rotation_euler = target_obj.matrix_world.to_euler()
        return True
    return False


class AUX_OT_snap_cursor_to_target(bpy.types.Operator):
    """Snap 3D Cursor to the location and rotation of the active target Bone or Object"""
    bl_idname = "anim.aux_snap_cursor_to_target"
    bl_label = "Snap 3D Cursor to Target"
    bl_description = "Snaps the 3D Cursor to the location and rotation of the active target Bone or Object"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        target_info = get_active_target_info(context)
        if target_info["mode"] == "NONE":
            self.report({"WARNING"}, "No active Object or Pose Bone selected to snap 3D Cursor to!")
            return {"CANCELLED"}
        success = snap_cursor_to_target_helper(context, target_info)
        if success:
            label = f"Bone '{target_info['name']}'" if target_info["mode"] == "BONE" else f"Object '{target_info['name']}'"
            self.report({"INFO"}, f"3D Cursor snapped to {label}")
            return {"FINISHED"}
        else:
            self.report({"WARNING"}, "Could not snap 3D Cursor to target.")
            return {"CANCELLED"}


class AUX_OT_generate_visual_anim(bpy.types.Operator):
    """Generate animation keyframes directly from drawn 3D annotations (Trajectory + Timing Ticks)"""
    bl_idname = "anim.aux_generate_visual_anim"
    bl_label = "Generate Animation"
    bl_description = "Generates animation keyframes from drawn 3D annotations (Trajectory + Timing Ticks)"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        settings = context.scene.aux_anim_graph_settings
        target_info = get_active_target_info(context)

        if target_info["mode"] == "NONE":
            self.report({"ERROR"}, "Please select an active Object or Armature Pose Bone!")
            return {"CANCELLED"}

        all_strokes = gather_all_annotation_strokes(context)
        if not all_strokes:
            self.report(
                {"ERROR"},
                "No visible annotation strokes found! Please draw a trajectory in 3D (hold D key)."
            )
            return {"CANCELLED"}

        all_strokes.sort(key=lambda s: s["length"], reverse=True)
        trajectory_info = all_strokes[0]
        traj_points = trajectory_info["points"]
        total_traj_len = trajectory_info["length"]

        if len(traj_points) < 2 or total_traj_len < 1e-4:
            self.report({"ERROR"}, "Main trajectory stroke is invalid or too short.")
            return {"CANCELLED"}

        cum_dists = calculate_cumulative_distances(traj_points)
        kd_tree = build_trajectory_kdtree(traj_points)

        marker_strokes = all_strokes[1:]
        raw_markers = []

        for m in marker_strokes:
            proj_data = project_marker_onto_trajectory(
                m["points"],
                traj_points,
                cum_dists,
                kd_tree=kd_tree,
                max_distance=settings.max_proximity,
            )
            if proj_data is not None:
                raw_markers.append(proj_data)

        raw_markers.sort(key=lambda x: x["distance_along_path"])

        filtered_markers = []
        min_spacing = total_traj_len * 0.005
        for m in raw_markers:
            if not filtered_markers:
                filtered_markers.append(m)
            else:
                dist_diff = m["distance_along_path"] - filtered_markers[-1]["distance_along_path"]
                if dist_diff >= min_spacing:
                    filtered_markers.append(m)

        keyframe_positions = []

        start_needed = settings.include_endpoints
        if filtered_markers and filtered_markers[0]["distance_along_path"] < (total_traj_len * 0.02):
            start_needed = False

        if start_needed:
            keyframe_positions.append({
                "distance": 0.0,
                "point": traj_points[0].copy(),
                "is_marker": False,
            })

        for m in filtered_markers:
            keyframe_positions.append({
                "distance": m["distance_along_path"],
                "point": m["projected_point"].copy(),
                "is_marker": True,
            })

        end_needed = settings.include_endpoints
        if filtered_markers and (total_traj_len - filtered_markers[-1]["distance_along_path"]) < (total_traj_len * 0.02):
            end_needed = False

        if end_needed or len(keyframe_positions) < 2:
            keyframe_positions.append({
                "distance": total_traj_len,
                "point": traj_points[-1].copy(),
                "is_marker": False,
            })

        keyframe_positions.sort(key=lambda k: k["distance"])

        num_keyframes = len(keyframe_positions)
        if num_keyframes < 2:
            self.report({"WARNING"}, "At least 2 keyframe points are required for animation.")
            return {"CANCELLED"}

        start_frame = context.scene.frame_current if settings.use_current_frame else settings.start_frame
        calculated_frames = []

        if settings.timing_mode == "STEP":
            step = settings.frame_step
            for i in range(num_keyframes):
                calculated_frames.append(start_frame + (i * step))
        else:
            total_dur = settings.total_duration
            for i in range(num_keyframes):
                fraction = i / float(num_keyframes - 1)
                f = start_frame + round(fraction * total_dur)
                if calculated_frames and f <= calculated_frames[-1]:
                    f = calculated_frames[-1] + 1
                calculated_frames.append(f)

        if settings.clear_existing:
            if target_info["mode"] == "BONE":
                arm_obj = target_info["armature"]
                if arm_obj.animation_data and arm_obj.animation_data.action:
                    d_prefix = f'pose.bones["{target_info["name"]}"]'
                    all_fc = get_action_fcurves_universal(arm_obj.animation_data.action)
                    for fc in list(all_fc):
                        if d_prefix in fc.data_path:
                            try:
                                arm_obj.animation_data.action.fcurves.remove(fc)
                            except Exception:
                                pass
            else:
                active_obj = target_info["object"]
                if active_obj.animation_data and active_obj.animation_data.action:
                    action = active_obj.animation_data.action
                    all_fc = get_action_fcurves_universal(action)
                    for fc in list(all_fc):
                        if fc.data_path in {"location", "rotation_euler", "rotation_quaternion", "rotation_axis_angle"}:
                            try:
                                action.fcurves.remove(fc)
                            except Exception:
                                pass

        if target_info["mode"] == "BONE":
            arm_obj = target_info["armature"]
            p_bone = target_info["pose_bone"]

            for i, k_data in enumerate(keyframe_positions):
                current_f = calculated_frames[i]
                pos = k_data["point"]
                s_dist = k_data["distance"]

                _, tangent = interpolate_position_on_trajectory(traj_points, cum_dists, s_dist)

                set_pose_bone_world_transform(
                    p_bone,
                    arm_obj,
                    pos,
                    tangent=tangent,
                    forward_axis=settings.forward_axis,
                    up_axis=settings.up_axis,
                    align_rotation=settings.align_rotation,
                )

                p_bone.keyframe_insert(data_path="location", frame=current_f)
                if settings.align_rotation:
                    if p_bone.rotation_mode == "QUATERNION":
                        p_bone.keyframe_insert(data_path="rotation_quaternion", frame=current_f)
                    elif p_bone.rotation_mode == "AXIS_ANGLE":
                        p_bone.keyframe_insert(data_path="rotation_axis_angle", frame=current_f)
                    else:
                        p_bone.keyframe_insert(data_path="rotation_euler", frame=current_f)

            apply_interpolation_to_keyframes(
                arm_obj,
                calculated_frames,
                settings.interpolation,
                data_path_filter=f'pose.bones["{p_bone.name}"]',
            )

        else:
            active_obj = target_info["object"]
            for i, k_data in enumerate(keyframe_positions):
                current_f = calculated_frames[i]
                pos = k_data["point"]
                s_dist = k_data["distance"]

                active_obj.location = pos
                active_obj.keyframe_insert(data_path="location", frame=current_f)

                if settings.align_rotation:
                    _, tangent = interpolate_position_on_trajectory(traj_points, cum_dists, s_dist)
                    if tangent.length > 1e-5:
                        track_code = settings.forward_axis
                        up_code = settings.up_axis
                        try:
                            rot_quat = tangent.to_track_quat(track_code, up_code)
                        except Exception:
                            rot_quat = tangent.to_track_quat("Y", "Z")

                        if active_obj.rotation_mode == "QUATERNION":
                            active_obj.rotation_quaternion = rot_quat
                            active_obj.keyframe_insert(data_path="rotation_quaternion", frame=current_f)
                        elif active_obj.rotation_mode == "AXIS_ANGLE":
                            active_obj.rotation_axis_angle = rot_quat.to_axis_angle()
                            active_obj.keyframe_insert(data_path="rotation_axis_angle", frame=current_f)
                        else:
                            active_obj.rotation_euler = rot_quat.to_euler(active_obj.rotation_mode)
                            active_obj.keyframe_insert(data_path="rotation_euler", frame=current_f)

            apply_interpolation_to_keyframes(active_obj, calculated_frames, settings.interpolation)

        if calculated_frames:
            last_frame = calculated_frames[-1]

            if settings.update_timeline_range:
                context.scene.frame_start = min(context.scene.frame_start, calculated_frames[0])
                context.scene.frame_end = max(context.scene.frame_end, last_frame)

            if getattr(settings, "jump_to_last_frame", True):
                context.scene.frame_set(last_frame)
            elif settings.update_timeline_range:
                context.scene.frame_set(calculated_frames[0])

            if getattr(settings, "auto_snap_cursor", True):
                context.view_layer.update()
                snap_cursor_to_target_helper(context, target_info)

            if getattr(settings, "auto_use_current_frame", True):
                settings.use_current_frame = True

        graph_name = get_motion_graph_name_for_target(target_info)
        graph_obj = bpy.data.objects.get(graph_name)
        if graph_obj:
            pts_world = [k["point"] for k in keyframe_positions]
            build_motion_graph_curve(context, target_info, pts_world, calculated_frames, tick_size=settings.graph_tick_size)

        mark_previous_annotations_red(context, all_strokes)

        msg = (
            f"Success! {num_keyframes} keyframes generated on '{target_info['full_label']}'. "
            f"({len(filtered_markers)} markers, Frames: {calculated_frames[0]} - {calculated_frames[-1]})"
        )
        self.report({"INFO"}, msg)
        return {"FINISHED"}


class AUX_OT_create_motion_graph(bpy.types.Operator):
    """Create a 3D Bezier curve graph from annotations or target animation and enter Edit Mode"""
    bl_idname = "anim.aux_create_motion_graph"
    bl_label = "Create / Edit Motion Graph"
    bl_description = "Generate an editable 3D Bezier curve motion graph from annotations or keyframes"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        active_obj = context.active_object
        settings = context.scene.aux_anim_graph_settings

        if active_obj and active_obj.name.startswith("VAG_Graph_"):
            if context.mode != "EDIT":
                bpy.ops.object.mode_set(mode="EDIT")
            return {"FINISHED"}

        target_info = get_active_target_info(context)
        if target_info["mode"] == "NONE":
            self.report({"ERROR"}, "Please select an active Object or Armature Pose Bone!")
            return {"CANCELLED"}

        graph_name = get_motion_graph_name_for_target(target_info)
        existing_graph = bpy.data.objects.get(graph_name)

        if existing_graph:
            if context.mode != "OBJECT":
                bpy.ops.object.mode_set(mode="OBJECT")
            if settings.auto_sync_annotation:
                try:
                    sync_motion_graph_to_annotations(context, existing_graph)
                except Exception as e:
                    print(f"[Auxnimate] Annotation sync notice: {e}")
            safe_set_active_object(context, existing_graph, select=True, mode="EDIT")
            self.report({"INFO"}, f"Existing '{graph_name}' entered Edit Mode.")
            return {"FINISHED"}

        all_strokes = gather_all_annotation_strokes(context)
        positions = []
        frames = []

        if all_strokes:
            bpy.ops.anim.aux_generate_visual_anim()
            mark_previous_annotations_red(context, all_strokes)
            if target_info["mode"] == "BONE":
                positions, frames = extract_keyframe_data_from_bone(context.scene, target_info["armature"], target_info["name"])
            else:
                positions, frames = extract_keyframe_data_from_object(target_info["object"])

        if not positions:
            if target_info["mode"] == "BONE":
                positions, frames = extract_keyframe_data_from_bone(context.scene, target_info["armature"], target_info["name"])
            else:
                positions, frames = extract_keyframe_data_from_object(target_info["object"])

        if not positions or len(positions) < 2:
            self.report(
                {"ERROR"},
                f"Cannot create Motion Graph! Draw a trajectory in 3D (D key) for '{target_info['full_label']}' or select an animated target."
            )
            return {"CANCELLED"}

        graph_obj = build_motion_graph_curve(
            context,
            target_info,
            positions,
            frames,
            tick_size=settings.graph_tick_size,
        )

        if context.mode != "OBJECT":
            try:
                bpy.ops.object.mode_set(mode="OBJECT")
            except Exception:
                pass

        if settings.auto_sync_annotation:
            try:
                sync_motion_graph_to_annotations(context, graph_obj)
            except Exception as e:
                print(f"[Auxnimate] Annotation sync notice: {e}")

        safe_set_active_object(context, graph_obj, select=True, mode="EDIT")

        self.report({"INFO"}, f"'{graph_name}' created! Edit points and handles in the 3D Viewport.")
        return {"FINISHED"}


class AUX_OT_apply_motion_graph(bpy.types.Operator):
    """Apply 3D Motion Graph edits to target animation keyframes"""
    bl_idname = "anim.aux_apply_motion_graph"
    bl_label = "Apply to Animation"
    bl_description = "Transfer 3D Motion Graph curve and handle edits to target animation keyframes"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        active_obj = context.active_object
        graph_obj = None

        if active_obj and active_obj.name.startswith("VAG_Graph_"):
            graph_obj = active_obj
        else:
            target_info = get_active_target_info(context)
            if target_info["mode"] != "NONE":
                graph_obj = bpy.data.objects.get(get_motion_graph_name_for_target(target_info))

        if not graph_obj:
            helpers = [o for o in bpy.data.objects if o.name.startswith("VAG_Graph_")]
            if helpers:
                graph_obj = helpers[0]

        if not graph_obj:
            self.report({"ERROR"}, "No active Motion Graph (VAG_Graph_*) found in scene!")
            return {"CANCELLED"}

        if context.mode != "OBJECT":
            bpy.ops.object.mode_set(mode="OBJECT")

        success = apply_motion_graph_to_target(context, graph_obj)
        if success:
            rebuild_graph_ticks(graph_obj, tick_size=context.scene.aux_anim_graph_settings.graph_tick_size)
            sync_motion_graph_to_annotations(context, graph_obj)
            target_name = graph_obj.get("vag_target_name", "Target")
            target_type = graph_obj.get("vag_target_type", "OBJECT")
            label = f"Bone '{target_name}'" if target_type == "BONE" else f"Object '{target_name}'"
            self.report({"INFO"}, f"Motion Graph successfully applied to {label} animation!")
            return {"FINISHED"}
        else:
            self.report({"ERROR"}, "Failed to apply Motion Graph to target animation.")
            return {"CANCELLED"}


class AUX_OT_sync_to_annotations(bpy.types.Operator):
    """Convert 3D Motion Graph curve and timing ticks into 3D Viewport annotations"""
    bl_idname = "anim.aux_sync_to_annotations"
    bl_label = "Convert Graph to Annotations"
    bl_description = "Display the edited 3D Motion Graph curve and timing markers as 3D Viewport annotations"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        active_obj = context.active_object
        graph_obj = None

        if active_obj and active_obj.name.startswith("VAG_Graph_"):
            graph_obj = active_obj
        else:
            target_info = get_active_target_info(context)
            if target_info["mode"] != "NONE":
                graph_obj = bpy.data.objects.get(get_motion_graph_name_for_target(target_info))

        if not graph_obj:
            helpers = [o for o in bpy.data.objects if o.name.startswith("VAG_Graph_")]
            if helpers:
                graph_obj = helpers[0]

        if not graph_obj:
            self.report({"ERROR"}, "No active Motion Graph (VAG_Graph_*) found in scene!")
            return {"CANCELLED"}

        if context.mode != "OBJECT":
            bpy.ops.object.mode_set(mode="OBJECT")

        success = sync_motion_graph_to_annotations(context, graph_obj)
        if success:
            self.report({"INFO"}, "Motion Graph successfully synced to 3D annotations!")
            return {"FINISHED"}
        else:
            self.report({"ERROR"}, "Failed to sync Motion Graph to annotations.")
            return {"CANCELLED"}


class AUX_OT_clear_graph_annotations(bpy.types.Operator):
    """Clear generated graph annotations (VAG_Trajectory, VAG_Markers, VAG_Source_*)"""
    bl_idname = "anim.aux_clear_graph_annotations"
    bl_label = "Clear Graph Annotations"
    bl_description = "Delete generated red trajectory and marker annotations while preserving active user drawing layers"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        cleared_layers = 0
        target_prefixes = ("VAG_", "Auxnimate_")

        for ann in list(bpy.data.annotations):
            for layer in list(ann.layers):
                if any(layer.info.startswith(p) for p in target_prefixes):
                    for f in list(layer.frames):
                        layer.frames.remove(f)
                    ann.layers.remove(layer)
                    cleared_layers += 1
            if len(ann.layers) == 0:
                bpy.data.annotations.remove(ann)

        self.report({"INFO"}, f"Cleared {cleared_layers} graph annotation layer(s). User drawing layers preserved.")
        return {"FINISHED"}


class AUX_OT_delete_motion_graph(bpy.types.Operator):
    """Delete 3D Motion Graph curve and restore target selection"""
    bl_idname = "anim.aux_delete_motion_graph"
    bl_label = "Delete Motion Graph"
    bl_description = "Remove 3D Motion Graph curve object and restore active selection to original target"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        if context.mode != "OBJECT":
            try:
                bpy.ops.object.mode_set(mode="OBJECT")
            except Exception:
                pass

        helpers = [o for o in bpy.data.objects if o.name.startswith("VAG_Graph_")]
        if not helpers:
            self.report({"INFO"}, "No Motion Graph found to delete.")
            return {"CANCELLED"}

        restore_obj = None
        restore_bone = None
        restore_mode = "OBJECT"

        for o in helpers:
            target_type = o.get("vag_target_type", "OBJECT")
            target_owner = o.get("vag_target_owner", "")
            target_name = o.get("vag_target_name", "")

            if target_type == "BONE" and target_owner:
                arm = bpy.data.objects.get(target_owner)
                if arm:
                    restore_obj = arm
                    restore_bone = target_name
                    restore_mode = "POSE"
            elif target_name:
                obj = bpy.data.objects.get(target_name)
                if obj:
                    restore_obj = obj
                    restore_mode = "OBJECT"

            curve_data = o.data
            bpy.data.objects.remove(o, do_unlink=True)
            if curve_data and curve_data.users == 0:
                bpy.data.curves.remove(curve_data)

        if not restore_obj:
            last_name = context.scene.get("vag_last_target_name", "")
            if last_name:
                restore_obj = bpy.data.objects.get(last_name)
                if context.scene.get("vag_last_target_type") == "BONE":
                    restore_mode = "POSE"
                    restore_bone = context.scene.get("vag_last_target_bone", "")

        if restore_obj:
            safe_set_active_object(context, restore_obj, select=True, mode=restore_mode, bone_name=restore_bone)
            context.scene.aux_properties_tab_active = True

        global _vag_prev_mode, _vag_last_state_cache, _vag_known_graphs
        _vag_prev_mode.clear()
        _vag_last_state_cache.clear()
        _vag_known_graphs.clear()

        for window in context.window_manager.windows:
            for area in window.screen.areas:
                if area.type == 'PROPERTIES':
                    if hasattr(area.spaces.active, "context"):
                        try:
                            target_tab = 'BONE' if restore_mode == 'POSE' else 'OBJECT'
                            prop = area.spaces.active.bl_rna.properties['context']
                            if target_tab in [it.identifier for it in prop.enum_items]:
                                area.spaces.active.context = target_tab
                        except Exception:
                            pass
                    area.tag_redraw()

        self.report({"INFO"}, f"Removed {len(helpers)} Motion Graph(s) and restored target selection.")
        return {"FINISHED"}


class AUX_OT_clear_annotations(bpy.types.Operator):
    """Clear all annotations in the scene"""
    bl_idname = "anim.aux_clear_annotations"
    bl_label = "Clear All Annotations"
    bl_description = "Delete all annotation frames and layers in the scene"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        cleared_count = 0
        for ann in list(bpy.data.annotations):
            for layer in list(ann.layers):
                for f in list(layer.frames):
                    layer.frames.remove(f)
                    cleared_count += 1
            if len(ann.layers) == 0:
                bpy.data.annotations.remove(ann)

        self.report({"INFO"}, f"All annotations cleared ({cleared_count} frame(s) removed).")
        return {"FINISHED"}


def draw_auxnimate_ui(layout, context):
    """Auxnimate Tools primary user interface (Properties Editor panel)"""
    scene = context.scene
    settings = getattr(scene, "aux_anim_graph_settings", None)
    if not settings:
        layout.label(text="Settings failed to load!", icon="ERROR")
        return

    target_info = get_active_target_info(context)

    box_target = layout.box()
    row_mode = box_target.row(align=True)
    row_mode.prop(settings, "target_mode", text="Target")

    row_info = box_target.row(align=True)
    if target_info["mode"] == "BONE":
        row_info.label(text=target_info["full_label"], icon="BONE_DATA")
        arm_obj = target_info["armature"]
        if arm_obj and arm_obj.pose:
            row_bone = box_target.row(align=True)
            row_bone.prop_search(settings, "target_bone", arm_obj.pose, "bones", text="Target Bone", icon="BONE_DATA")
        row_cursor = box_target.row(align=True)
        row_cursor.operator("anim.aux_snap_cursor_to_target", text="Snap Cursor to Bone", icon="CURSOR")
    elif target_info["mode"] == "OBJECT":
        row_info.label(text=target_info["full_label"], icon="OBJECT_DATA")
        row_cursor = box_target.row(align=True)
        row_cursor.operator("anim.aux_snap_cursor_to_target", text="Snap Cursor to Object", icon="CURSOR")
    else:
        row_info.alert = True
        row_info.label(text=target_info["full_label"], icon="ERROR")

    stroke_count = 0
    try:
        all_strokes = gather_all_annotation_strokes(context)
        stroke_count = len(all_strokes)
    except Exception:
        stroke_count = 0

    status_row = box_target.row(align=True)
    if stroke_count == 0:
        status_row.label(text="Drawing: None (Draw in 3D with D key)", icon="GREASEPENCIL")
    elif stroke_count == 1:
        status_row.label(text="Drawing: 1 Trajectory (No timing ticks)", icon="GP_SELECT_STROKES")
    else:
        status_row.label(text=f"Drawing: 1 Trajectory + {stroke_count - 1} Timing Ticks", icon="CHECKMARK")

    box_timing = layout.box()
    box_timing.label(text="Timing & Motion", icon="TIME")

    col_timing = box_timing.column(align=True)
    col_timing.prop(settings, "timing_mode", expand=True)

    if settings.timing_mode == "STEP":
        col_timing.prop(settings, "frame_step", icon="FORWARD")
    else:
        col_timing.prop(settings, "total_duration", icon="TIME")

    row_start = col_timing.row(align=True)
    row_start.prop(settings, "start_frame", icon="PLAY")
    row_start.prop(settings, "use_current_frame", text="Current Frame")

    col_interp = box_timing.column(align=True)
    row_interp = col_interp.row(align=True)
    row_interp.label(text="Interpolation:", icon="IPO_BEZIER")
    row_interp.prop(settings, "interpolation", text="")

    col_rot = box_timing.column(align=True)
    col_rot.prop(settings, "align_rotation", text="Align Rotation to Path", icon="ORIENTATION_GIMBAL")
    if settings.align_rotation:
        row_axes = col_rot.row(align=True)
        row_axes.prop(settings, "forward_axis", text="Forward")
        row_axes.prop(settings, "up_axis", text="Up")

    box_graph = layout.box()
    box_graph.label(text="3D Motion Graph", icon="CURVE_PATH")

    active_graph_name = get_motion_graph_name_for_target(target_info)
    graph_obj = bpy.data.objects.get(active_graph_name)

    if graph_obj:
        info_row = box_graph.row(align=True)
        info_row.label(text=f"Active Graph: {graph_obj.name}", icon="CHECKMARK")

        row_graph_ops = box_graph.row(align=True)
        row_graph_ops.scale_y = 1.3
        row_graph_ops.operator("anim.aux_create_motion_graph", text="Edit in 3D (Edit Mode)", icon="EDITMODE_HLT")
        row_graph_ops.operator("anim.aux_apply_motion_graph", text="Apply to Animation", icon="FILE_REFRESH")

        row_sync = box_graph.row(align=True)
        row_sync.operator("anim.aux_sync_to_annotations", text="Draw to Annotations", icon="GREASEPENCIL")
        row_sync.operator("anim.aux_clear_graph_annotations", text="Clear Graph Drawing", icon="X")

        row_del = box_graph.row(align=True)
        row_del.operator("anim.aux_delete_motion_graph", text="Delete Motion Graph", icon="TRASH")

    else:
        col_create = box_graph.column(align=True)
        col_create.scale_y = 1.3
        col_create.operator("anim.aux_create_motion_graph", text="Create / Edit Motion Graph", icon="CURVE_PATH")

        row_annot_tools = box_graph.row(align=True)
        row_annot_tools.operator("anim.aux_sync_to_annotations", text="Draw to Annotations", icon="GREASEPENCIL")
        row_annot_tools.operator("anim.aux_clear_graph_annotations", text="Clear Graph Drawing", icon="X")

        if [o for o in bpy.data.objects if o.name.startswith("VAG_Graph_")]:
            box_graph.operator("anim.aux_delete_motion_graph", text="Delete All Graphs", icon="TRASH")

    col_graph_sync = box_graph.column(align=True)
    col_graph_sync.prop(settings, "auto_sync_graph")
    col_graph_sync.prop(settings, "auto_sync_annotation")

    row_cols = col_graph_sync.row(align=True)
    row_cols.prop(settings, "trajectory_color", text="Path Color")
    row_cols.prop(settings, "marker_color", text="Tick Color")

    col_graph_sync.prop(settings, "graph_tick_size", text="Tick Size")

    box_adv = layout.box()
    box_adv.label(text="Advanced Settings", icon="PREFERENCES")
    col_adv = box_adv.column(align=True)
    col_adv.prop(settings, "max_proximity", text="Proximity Threshold", icon="ARROW_LEFTRIGHT")
    col_adv.prop(settings, "include_endpoints")
    col_adv.prop(settings, "clear_existing")
    col_adv.prop(settings, "update_timeline_range")
    col_adv.prop(settings, "jump_to_last_frame")
    col_adv.prop(settings, "auto_snap_cursor")
    col_adv.prop(settings, "auto_use_current_frame")

    layout.separator()
    btn_col = layout.column(align=True)
    btn_col.scale_y = 1.6
    btn_col.operator("anim.aux_generate_visual_anim", icon="ACTION", text="Generate Animation")

    row_tools = layout.row(align=True)
    row_tools.scale_y = 1.1
    row_tools.operator("anim.aux_clear_annotations", icon="TRASH", text="Clear All Annotations")


class AUX_OT_toggle_properties_tab(bpy.types.Operator):
    """Toggle the Auxnimate Tools panel in Properties (Shift+F7)"""
    bl_idname = "anim.aux_toggle_properties_tab"
    bl_label = "Auxnimate Tools"
    bl_description = "Toggle the Auxnimate Tools panel in the Properties sidebar"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        scene = context.scene
        view = getattr(context, "space_data", None)

        if not context.active_object:
            target_obj = None
            last_target_name = scene.get("vag_last_target_name", "")
            if last_target_name:
                target_obj = bpy.data.objects.get(last_target_name)
            if not target_obj:
                for obj in scene.objects:
                    if not obj.name.startswith("VAG_Graph_") and obj.type in {"MESH", "ARMATURE", "CURVE", "EMPTY"}:
                        target_obj = obj
                        break
            if target_obj:
                mode = "POSE" if (scene.get("vag_last_target_type") == "BONE" and target_obj.type == "ARMATURE") else "OBJECT"
                safe_set_active_object(context, target_obj, select=True, mode=mode, bone_name=scene.get("vag_last_target_bone"))

        target_ctx = 'BONE' if getattr(context, "mode", "") == 'POSE' else 'OBJECT'

        if view and hasattr(view, "context") and view.context != target_ctx:
            try:
                prop = view.bl_rna.properties['context']
                if target_ctx in [it.identifier for it in prop.enum_items]:
                    view.context = target_ctx
                elif "OBJECT" in [it.identifier for it in prop.enum_items]:
                    view.context = "OBJECT"
            except Exception:
                pass
            scene.aux_properties_tab_active = True
            self.report({"INFO"}, f"Auxnimate Tools opened under {target_ctx.capitalize()} properties.")
        else:
            current_state = getattr(scene, "aux_properties_tab_active", True)
            new_state = not current_state
            scene.aux_properties_tab_active = new_state
            if new_state:
                self.report({"INFO"}, "Auxnimate Tools panel opened.")
            else:
                self.report({"INFO"}, "Auxnimate Tools panel hidden.")

        for window in context.window_manager.windows:
            for area in window.screen.areas:
                if area.type == 'PROPERTIES':
                    area.tag_redraw()

        return {'FINISHED'}


class AuxnimatePropertiesPanelBase:
    bl_space_type = "PROPERTIES"
    bl_region_type = "WINDOW"
    bl_label = "Auxnimate Tools"
    bl_order = 10000

    @classmethod
    def poll(cls, context):
        return getattr(context.scene, "aux_properties_tab_active", True)

    def draw(self, context):
        draw_auxnimate_ui(self.layout, context)


class OBJECT_PT_auxnimate_tools(AuxnimatePropertiesPanelBase, bpy.types.Panel):
    """Auxnimate Tools panel placed at the bottom of the Object properties tab"""
    bl_idname = "OBJECT_PT_auxnimate_tools"
    bl_context = "object"


class BONE_PT_auxnimate_tools(AuxnimatePropertiesPanelBase, bpy.types.Panel):
    """Auxnimate Tools panel placed at the bottom of the Bone properties tab (Pose Mode)"""
    bl_idname = "BONE_PT_auxnimate_tools"
    bl_context = "bone"


_orig_properties_nav_bar_draw = None


def aux_properties_nav_bar_draw(self, context):
    """
    Draws the vertical tab bar in Properties (Shift+F7):
    1. Standard tabs (prop_tabs_enum)
    2. Auxnimate Tools tab button (Right above Visibility button)
    3. Visibility popover button
    """
    layout = self.layout
    view = context.space_data

    layout.scale_x = 1.4
    layout.scale_y = 1.4
    if getattr(view, "search_filter", None):
        layout.prop_tabs_enum(
            view, "context", data_highlight=view,
            property_highlight="tab_search_results", icon_only=True,
        )
    else:
        layout.prop_tabs_enum(view, "context", icon_only=True)

    target_ctx = 'BONE' if getattr(context, "mode", "") == 'POSE' else 'OBJECT'
    is_active = (getattr(view, "context", "") == target_ctx) and getattr(context.scene, "aux_properties_tab_active", True)
    sub_aux = layout.row(align=True)
    sub_aux.alignment = 'CENTER'
    sub_aux.operator("anim.aux_toggle_properties_tab", icon="ANIM", text="", depress=is_active)

    sub = layout.row(align=True)
    sub.alignment = 'CENTER'
    sub.emboss = 'NONE'
    sub.scale_x = 0.8
    sub.scale_y = 0.8
    sub.separator(factor=0.7)
    sub.popover(panel="PROPERTIES_PT_visibility", text="")
    try:
        import bl_ui.space_properties as sp
        sub.active = sp.has_hidden_tabs(view)
    except Exception:
        pass


AUX_PT_properties_popover = OBJECT_PT_auxnimate_tools
PROPERTIES_PT_auxnimate_tools = OBJECT_PT_auxnimate_tools
AUX_PT_visual_anim_graph = OBJECT_PT_auxnimate_tools

classes = (
    AUX_AnimGraphSettings,
    AUX_OT_toggle_properties_tab,
    AUX_OT_snap_cursor_to_target,
    AUX_OT_generate_visual_anim,
    AUX_OT_create_motion_graph,
    AUX_OT_apply_motion_graph,
    AUX_OT_sync_to_annotations,
    AUX_OT_clear_graph_annotations,
    AUX_OT_delete_motion_graph,
    AUX_OT_clear_annotations,
    OBJECT_PT_auxnimate_tools,
    BONE_PT_auxnimate_tools,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.aux_anim_graph_settings = bpy.props.PointerProperty(type=AUX_AnimGraphSettings)
    bpy.types.Scene.aux_properties_tab_active = bpy.props.BoolProperty(
        name="Auxnimate Tab Active",
        description="Toggle visibility of the Auxnimate Tools panel in Properties",
        default=True,
    )

    try:
        import bl_ui.space_properties as sp
        global _orig_properties_nav_bar_draw
        if _orig_properties_nav_bar_draw is None:
            _orig_properties_nav_bar_draw = sp.PROPERTIES_PT_navigation_bar.draw
        sp.PROPERTIES_PT_navigation_bar.draw = aux_properties_nav_bar_draw
    except Exception as e:
        print("Auxnimate: Navigation bar registration skipped:", e)

    if update_vag_realtime not in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.append(update_vag_realtime)


def unregister():
    if update_vag_realtime in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.remove(update_vag_realtime)

    try:
        import bl_ui.space_properties as sp
        global _orig_properties_nav_bar_draw
        if _orig_properties_nav_bar_draw is not None:
            sp.PROPERTIES_PT_navigation_bar.draw = _orig_properties_nav_bar_draw
            _orig_properties_nav_bar_draw = None
    except Exception:
        pass

    for cls in reversed(classes):
        try:
            bpy.utils.unregister_class(cls)
        except Exception:
            pass

    if hasattr(bpy.types.Scene, "aux_properties_tab_active"):
        del bpy.types.Scene.aux_properties_tab_active

    if hasattr(bpy.types.Scene, "aux_anim_graph_settings"):
        del bpy.types.Scene.aux_anim_graph_settings


if __name__ == "__main__":
    register()
