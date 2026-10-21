"""Build the standalone Go1 + DM scene from the pinned, unmodified URDF."""
from pathlib import Path
import copy
import math
import xml.etree.ElementTree as ET

HERE = Path(__file__).resolve().parent
MODELS = HERE.parent


def quaternion(rpy):
    r, p, y = (float(v) / 2 for v in rpy.split())
    cr, cp, cy = math.cos(r), math.cos(p), math.cos(y)
    sr, sp, sy = math.sin(r), math.sin(p), math.sin(y)
    return ' '.join(map(str, (cr*cp*cy+sr*sp*sy, sr*cp*cy-cr*sp*sy,
                              cr*sp*cy+sr*cp*sy, cr*cp*sy-sr*sp*cy)))


def origin(element):
    o = element.find('origin')
    return dict(pos=o.get('xyz', '0 0 0') if o is not None else '0 0 0',
                quat=quaternion(o.get('rpy', '0 0 0') if o is not None else '0 0 0'))


def build(source_path=None, *, write=True):
    source_path = Path(source_path) if source_path else MODELS/'go1/go1_scene_mppi.xml'
    root = ET.parse(source_path).getroot()
    root.set('model', 'Go1 with reBot B601-DM')
    root.find('size').set('memory', '16M')
    root.find('include').set('file', '../common.xml')
    inc = root.findall('include')[1]
    index = list(root).index(inc)
    root.remove(inc)
    go1 = ET.parse(source_path.parent/inc.get('file')).getroot()
    go1.find('compiler').set('meshdir', '.')
    for mesh in go1.findall('./asset/mesh'):
        mesh.set('file', '../go1/assets/' + mesh.get('file'))
    for child in go1:
        root.insert(index, copy.deepcopy(child))
        index += 1
    # Finer integration than the planner's 10 ms interval for the arm/gripper.
    option = root.find('option')
    if option is None:
        option = ET.SubElement(root, 'option')
    option.set('timestep', '0.002')
    option.set('integrator', 'implicitfast')
    defaults = ET.SubElement(root, 'default')
    dm = ET.SubElement(defaults, 'default', {'class': 'dm'})
    ET.SubElement(dm, 'joint', damping='0.15', armature='0.01', frictionloss='0.05')
    ET.SubElement(dm, 'geom', friction='0.7 0.01 0.001', condim='3', margin='0.001')
    asset = ET.SubElement(root, 'asset')
    urdf = ET.parse(MODELS/'rebot_dm/upstream/urdf/ReBot_Arm_DM.urdf').getroot()
    materials = {m.get('name'): m.find('color').get('rgba') for m in urdf.findall('material')}
    links = {l.get('name'): l for l in urdf.findall('link')}
    children = {}
    for joint in urdf.findall('joint'):
        children.setdefault(joint.find('parent').get('link'), []).append(joint)
    meshes = {}

    def mesh_asset(filename):
        if filename not in meshes:
            name = 'dm_mesh_' + str(len(meshes))
            meshes[filename] = name
            ET.SubElement(asset, 'mesh', name=name,
                          file='../rebot_dm/upstream/' + filename.removeprefix('../'))
        return meshes[filename]

    actuator = root.find('actuator')
    equality = ET.SubElement(root, 'equality')
    contact = ET.SubElement(root, 'contact')

    def add_link(name, parent, joint=None):
        attrs = origin(joint) if joint is not None else dict(pos='0 0 0.10')
        body = ET.SubElement(parent, 'body', name='dm_'+name, childclass='dm', **attrs)
        link = links[name]
        inertial = link.find('inertial')
        if inertial is not None:
            mass = float(inertial.find('mass').get('value'))
            # The upstream end_link lumps the fingers into its 0.5 kg mass.
            # Allocate 30 g per moving finger, preserving total assembly mass.
            if name == 'end_link':
                mass -= 0.06
            if inertial.find('origin').get('rpy', '0 0 0') != '0 0 0':
                raise ValueError('Rotate the inertia tensor before exporting nonzero inertial rpy')
            inertia = inertial.find('inertia')
            ET.SubElement(body, 'inertial', mass=str(mass), pos=origin(inertial)['pos'],
                          fullinertia=' '.join(inertia.get(k) for k in ('ixx','iyy','izz','ixy','ixz','iyz')))
        elif name.startswith('finger_'):
            ET.SubElement(body, 'inertial', pos='0 0 0', mass='0.03', diaginertia='0.00002 0.00002 0.00002')
        if joint is not None and joint.get('type') != 'fixed':
            limit = joint.find('limit')
            jname = 'dm_'+joint.get('name')
            ET.SubElement(body, 'joint', name=jname,
                          type='slide' if joint.get('type') == 'prismatic' else 'hinge',
                          axis=joint.find('axis').get('xyz'),
                          range=limit.get('lower')+' '+limit.get('upper'))
            if joint.find('mimic') is None:
                finger = joint.get('type') == 'prismatic'
                kp, kv = ('300', '5') if finger else ('55', '3')
                effort = limit.get('effort')
                ET.SubElement(actuator, 'general', name=jname, joint=jname,
                              biastype='affine', gainprm=kp, biasprm=f'0 -{kp} -{kv}',
                              ctrlrange=limit.get('lower')+' '+limit.get('upper'),
                              forcerange=f'-{effort} {effort}')
            else:
                ET.SubElement(equality, 'joint', joint1=jname, joint2='dm_finger_left', polycoef='0 -1 0 0 0')
        for visual in link.findall('visual'):
            filename = visual.find('geometry/mesh').get('filename')
            mat = visual.find('material')
            ET.SubElement(body, 'geom', type='mesh', mesh=mesh_asset(filename),
                          contype='0', conaffinity='0', group='2', density='0',
                          rgba=materials.get(mat.get('name'), '0.5 0.5 0.5 1'), **origin(visual))
        # Split finger collisions avoid the convex hull filling the grasp gap.
        if name.startswith('finger_'):
            side = 'left' if 'left' in name else 'right'
            files = [f'../meshes/mujoco_collision/{side}_finger_{part}.stl' for part in ('front','mid','rear')]
            files += [f'../meshes/shared/colored_{side}_finger_{part}.stl' for part in ('carriage_grey','travel_stop_yellow')]
            for filename in files:
                ET.SubElement(body, 'geom', type='mesh', mesh=mesh_asset(filename), group='3', density='0')
        else:
            for collision in link.findall('collision'):
                ET.SubElement(body, 'geom', type='mesh', mesh=mesh_asset(collision.find('geometry/mesh').get('filename')),
                              group='3', density='0', **origin(collision))
        if name == 'end_link':
            ET.SubElement(body, 'site', name='dm_tcp', pos='0 0 0', size='0.008', rgba='1 0 0 1')
        for child_joint in children.get(name, []):
            child_name = child_joint.find('child').get('link')
            add_link(child_name, body, child_joint)
            # Adjacent housings overlap in CAD; exclude only connected links.
            ET.SubElement(contact, 'exclude', body1='dm_'+name, body2='dm_'+child_name)
        return body

    trunk = root.find(".//body[@name='trunk']")
    mount = ET.SubElement(trunk, 'body', name='dm_mount', pos='0 0 0', childclass='dm')
    ET.SubElement(mount, 'inertial', pos='0 0 0.075', mass='0.2', diaginertia='0.0003 0.0003 0.0005')
    ET.SubElement(mount, 'geom', type='box', pos='0 0 0.075', size='0.09 0.065 0.025', rgba='0.25 0.25 0.25 1')
    add_link('base_link', mount)
    ET.SubElement(contact, 'exclude', body1='trunk', body2='dm_mount')
    ET.SubElement(contact, 'exclude', body1='dm_mount', body2='dm_base_link')
    # The URDF wrist housing hulls overlap by 0.2 mm at neutral.
    ET.SubElement(contact, 'exclude', body1='dm_link4', body2='dm_link6')
    # Appending the arm after all four legs preserves the Go1 joint ordering.
    root.find('./keyframe/key').set('qpos', root.find('./keyframe/key').get('qpos').replace('0 0 0.26', '0 0 0.28'))
    arm_home = '0 -1.57 -1.57 0 0 0 0.02 -0.02'
    for key in root.findall('./keyframe/key'):
        key.set('qpos', key.get('qpos')+' '+arm_home)
        key.set('ctrl', key.get('ctrl')+' 0 -1.57 -1.57 0 0 0 0.02')
    root.set('model', 'Go1 + B601-DM (generated; see README.md)')
    ET.indent(root)
    if write:
        ET.ElementTree(root).write(HERE/'scene.xml', encoding='utf-8', xml_declaration=True)
    return root


def load_model(source_path):
    """Mount DM on an existing task scene, preserving terrain and leg contacts."""
    import mujoco
    root = build(source_path, write=False)
    for element in root.iter():
        if 'file' in element.attrib:
            element.set('file', str((HERE / element.get('file')).resolve()))
    return mujoco.MjModel.from_xml_string(ET.tostring(root, encoding='unicode'))


if __name__ == '__main__':
    build()
