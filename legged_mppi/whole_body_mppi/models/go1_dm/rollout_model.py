"""Shared primitive DM model for prediction, simulation and rendering."""
import xml.etree.ElementTree as ET

import numpy as np

from whole_body_mppi.models.go1_dm.build_model import build, HERE

# Local link-frame capsule endpoints and radii, in metres. Conservative proxies
# for the arm housings, not a contact-accurate model for manipulation.
CAPSULES = {
    'link1': ('0 0 0.005 0 0 0.045', '0.035'),
    'link2': ('-0.015 0 -0.03 -0.25 0 -0.03', '0.03'),
    'link3': ('0 0 -0.03 0.24 -0.054 -0.03', '0.03'),
    'link4': ('0 0 -0.03 0.078 -0.0375 -0.03', '0.03'),
    'link5': ('0 0 0.01 0.028 0 0.04', '0.028'),
    'link6': ('0 0 0.01 0 0 0.065', '0.03'),
}


def load_rollout_model(source_path, timestep=0.01, *, rigid=False):
    """Preserve all six arm joints/inertias, weld fingers at their home pose."""
    import mujoco
    root = build(source_path, write=False)
    # Use Go1's collision defaults for the added arm proxies as well.
    go1_geom = root.find("./default/default[@class='go1']/geom")
    dm_geom = root.find("./default/default[@class='dm']/geom")
    dm_geom.attrib.update(go1_geom.attrib)
    for body in root.findall('.//body'):
        name = body.get('name', '')
        if not name.startswith('dm_'):
            continue
        for geom in list(body.findall('geom')):
            if name != 'dm_mount':
                body.remove(geom)
        link = name.removeprefix('dm_')
        if link in CAPSULES:
            endpoints, radius = CAPSULES[link]
            ET.SubElement(body, 'geom', type='capsule', fromto=endpoints, size=radius, density='0', group='3')
        elif link == 'base_link':
            ET.SubElement(body, 'geom', type='cylinder', pos='0 0 0.035', size='0.06 0.035', density='0', group='3')
        elif link == 'end_link':
            ET.SubElement(body, 'geom', type='box', pos='-0.045 0 0', size='0.045 0.035 0.025', density='0', group='3')
        elif link.startswith('finger_'):
            for joint in list(body.findall('joint')):
                body.remove(joint)
            body.set('pos', '0 0.02 0' if 'left' in link else '0 -0.02 0')
            ET.SubElement(body, 'geom', type='box', pos='-0.09 0 0', size='0.04 0.008 0.012', density='0', group='3')
    for actuator in root.findall('actuator'):
        for item in list(actuator):
            if item.get('joint', '').startswith('dm_finger_'):
                actuator.remove(item)
    for equality in root.findall('equality'):
        for item in list(equality):
            if item.get('joint1', '').startswith('dm_finger_'):
                equality.remove(item)
    for asset in root.findall('asset'):
        for item in list(asset):
            if item.get('name', '').startswith('dm_mesh_'):
                asset.remove(item)
    for key in root.findall('./keyframe/key'):
        key.set('qpos', ' '.join(key.get('qpos').split()[:-2]))
        key.set('ctrl', ' '.join(key.get('ctrl').split()[:-1]))
    # Match the source Go1 integrator, not the detailed assembly's override.
    source = mujoco.MjModel.from_xml_path(str(source_path))
    root.find('option').set('integrator', 'Euler')
    root.find('option').set('timestep', str(timestep))
    # Primitive visuals follow Go1's visual/collision group convention.
    for body in root.findall('.//body'):
        if not body.get('name', '').startswith('dm_'):
            continue
        for geom in list(body.findall('geom')):
            visual = ET.SubElement(body, 'geom', dict(geom.attrib))
            visual.set('group', '2')
            visual.set('contype', '0')
            visual.set('conaffinity', '0')
            visual.set('rgba', '0.73 0.84 0.12 1' if body.get('name') in ('dm_link2', 'dm_link3') else '0.25 0.28 0.27 1')
    for element in root.iter():
        if 'file' in element.attrib:
            element.set('file', str((HERE / element.get('file')).resolve()))
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding='unicode'))
    model.opt.integrator = source.opt.integrator
    return _rigid_payload(root, model) if rigid else model


def load_runtime_model(source_path, variant='full', timestep=0.01):
    """Load the same variant and numerical settings for both runtime paths."""
    if variant not in ('simple', 'full'):
        raise ValueError(f'Unknown DM model variant: {variant}')
    return load_rollout_model(source_path, timestep, rigid=variant == 'simple')


def _rigid_payload(root, articulated):
    """Aggregate the arm at home, preserving its composite mass and inertia."""
    import mujoco
    data = mujoco.MjData(articulated)
    mujoco.mj_resetDataKeyframe(articulated, data, articulated.key('home').id)
    mujoco.mj_forward(articulated, data)
    trunk_id = articulated.body('trunk').id
    rotation = data.xmat[trunk_id].reshape(3, 3)
    position = data.xpos[trunk_id]
    ids = [i for i in range(articulated.nbody) if articulated.body(i).name.startswith('dm_')]
    masses = articulated.body_mass[ids]
    centers = (data.xipos[ids] - position) @ rotation
    mass = masses.sum()
    center = np.sum(masses[:, None]*centers, axis=0)/mass
    inertia = np.zeros((3, 3))
    for i, m, c in zip(ids, masses, centers):
        orientation = rotation.T @ data.ximat[i].reshape(3, 3)
        delta = c-center
        inertia += orientation @ np.diag(articulated.body_inertia[i]) @ orientation.T
        inertia += m*(np.dot(delta, delta)*np.eye(3)-np.outer(delta, delta))
    trunk = root.find(".//body[@name='trunk']")
    trunk.remove(trunk.find("body[@name='dm_mount']"))
    payload = ET.SubElement(trunk, 'body', name='dm_payload', childclass='dm')
    def values(array):
        return ' '.join(format(float(v), '.17g') for v in np.asarray(array).ravel())
    # Supply principal axes explicitly to avoid the compiler's approximate
    # fullinertia diagonalization losing small off-diagonal terms.
    principal, axes = np.linalg.eigh(inertia)
    if np.linalg.det(axes) < 0:
        axes[:, 0] *= -1
    inertial_quat = np.empty(4)
    mujoco.mju_mat2Quat(inertial_quat, axes.ravel())
    ET.SubElement(payload, 'inertial', mass=str(mass), pos=values(center),
                  diaginertia=values(principal), quat=values(inertial_quat))
    types = {mujoco.mjtGeom.mjGEOM_BOX: ('box', 3),
             mujoco.mjtGeom.mjGEOM_CAPSULE: ('capsule', 2),
             mujoco.mjtGeom.mjGEOM_CYLINDER: ('cylinder', 2)}
    # Keep the same primitive silhouette and exterior collision envelope.
    for g in range(articulated.ngeom):
        if articulated.geom_bodyid[g] not in ids:
            continue
        kind, width = types[articulated.geom_type[g]]
        quat = np.empty(4)
        mujoco.mju_mat2Quat(quat, (rotation.T @ data.geom_xmat[g].reshape(3,3)).ravel())
        ET.SubElement(payload, 'geom', type=kind, size=values(articulated.geom_size[g, :width]),
                      pos=values(rotation.T @ (data.geom_xpos[g]-position)), quat=values(quat),
                      group=str(articulated.geom_group[g]), rgba=values(articulated.geom_rgba[g]),
                      contype=str(articulated.geom_contype[g]),
                      conaffinity=str(articulated.geom_conaffinity[g]), density='0')
    tcp = articulated.site('dm_tcp').id
    tcp_quat = np.empty(4)
    mujoco.mju_mat2Quat(tcp_quat, (rotation.T @ data.site_xmat[tcp].reshape(3,3)).ravel())
    ET.SubElement(payload, 'site', name='dm_tcp', pos=values(rotation.T @ (data.site_xpos[tcp]-position)),
                  quat=values(tcp_quat), size='0.008', rgba='1 0 0 1')
    for section in root.findall('actuator'):
        for item in list(section):
            if item.get('joint', '').startswith('dm_'):
                section.remove(item)
    for section in root.findall('contact'):
        for item in list(section):
            if item.get('body1', '').startswith('dm_') or item.get('body2', '').startswith('dm_'):
                section.remove(item)
        ET.SubElement(section, 'exclude', body1='trunk', body2='dm_payload')
    for key in root.findall('./keyframe/key'):
        key.set('qpos', ' '.join(key.get('qpos').split()[:19]))
        key.set('ctrl', ' '.join(key.get('ctrl').split()[:12]))
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding='unicode'))
    model.opt.integrator = articulated.opt.integrator
    return model
