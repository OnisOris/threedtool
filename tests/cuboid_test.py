import pytest
import numpy as np
from threedtool.core.cuboid import Cuboid
from threedtool.fmath.fmath import rot_x, rot_y, rot_z, rot_v


class TestCuboid:
    def test_create_default(self):
        cub = Cuboid()
        # Default center is at origin
        assert isinstance(cub.center, np.ndarray)
        assert cub.center.shape == (3,)
        assert np.allclose(cub.center, np.zeros(3))
        # Default size is ones
        assert np.allclose(cub.length_width_height, np.ones(3))
        # Default rotation is identity
        assert np.allclose(cub.rotation, np.eye(3))
        # Default color
        assert cub.color == "red"

    def test_length_width_height_properties(self):
        cub = Cuboid()
        # Test getters
        assert cub.length == 1.0
        assert cub.width == 1.0
        assert cub.height == 1.0
        # Test setters
        cub.length = 2.5
        cub.width = 3.5
        cub.height = 4.5
        assert np.allclose(cub.length_width_height, np.array([2.5, 3.5, 4.5]))
        assert cub.length == 2.5
        assert cub.width == 3.5
        assert cub.height == 4.5

    def test_get_vertices_default(self):
        cub = Cuboid()
        verts = cub.get_vertices()
        # Should return 8 vertices
        assert isinstance(verts, np.ndarray)
        assert verts.shape == (8, 3)
        # For unit cube centered at origin, vertices coords should be all combinations of ±0.5
        expected = np.array([[x, y, z] for x in (-0.5, 0.5)
                                          for y in (-0.5, 0.5)
                                          for z in (-0.5, 0.5)])
        # Sorting rows for comparison
        assert np.allclose(np.sort(verts, axis=0), np.sort(expected, axis=0))

    def test_vertices_base_and_axes(self):
        cub = Cuboid()
        base = cub.vertices_base
        # vertices_base should be first 4 vertices
        verts = cub.get_vertices()
        assert np.array_equal(base, verts[:4])
        # get_axes should return identity axes
        axes = cub.get_axes()
        assert np.allclose(axes, np.eye(3))

    def test_height_vector(self):
        length_width_height = np.ones((3,))
        cub = Cuboid(length_width_height=length_width_height)
        # height_vector = third axis * height
        expected = np.array([0, 0, 1]) * 1.0
        assert np.allclose(cub.height_vector, expected)

    @pytest.mark.parametrize("angle, rot_fn", [
        (np.pi/2, rot_x),
        (np.pi/4, rot_y),
        (np.pi/6, rot_z)
    ])
    def test_rotate_single_axis(self, angle, rot_fn):
        cub = Cuboid()
        original = cub.rotation.copy()
        # apply the rotation method
        if rot_fn == rot_x:
            cub.rotate_x(angle)
        elif rot_fn == rot_y:
            cub.rotate_y(angle)
        else:
            cub.rotate_z(angle)
        # rotation should equal original @ rot_fn(angle)
        expected = original @ rot_fn(angle)
        assert np.allclose(cub.rotation, expected)

    def test_rotate_v(self):
        cub = Cuboid()
        axis = np.array([1, 1, 0]) / np.sqrt(2)
        angle = np.pi / 3
        orig = cub.rotation.copy()
        cub.rotate_v(axis, angle)
        expected = orig @ rot_v(angle=angle, axis=axis)
        assert np.allclose(cub.rotation, expected)

    def test_rotate_euler(self):
        cub = Cuboid()
        alpha, beta, gamma = np.pi/3, np.pi/4, np.pi/6
        orig = cub.rotation.copy()
        cub.rotate_euler(alpha, beta, gamma)
        expected = rot_z(alpha) @ rot_x(beta) @ rot_z(gamma) @ orig
        assert np.allclose(cub.rotation, expected)

    def test_get_edges(self):
        cub = Cuboid()
        edges = cub.get_edges()
        assert isinstance(edges, list)
        assert len(edges) == 12
        # Check some sample edges
        assert (0, 1) in edges
        assert (6, 7) in edges

    def test_get_face_normals(self):
        cub = Cuboid()
        normals = cub.get_face_normals()
        # For unit cube aligned with axes, normals should include ±z, ±x, ±y
        expected_normals = [np.array([0, 0, 1]), np.array([0, 0, -1]),
                             np.array([1, 0, 0]), np.array([-1, 0, 0]),
                             np.array([0, 1, 0]), np.array([0, -1, 0])]
        # Normalize expected indeed unit vectors
        # Compare sets by direction ignoring order
        for en in expected_normals:
            assert any(np.allclose(en, n) for n in normals), f"Missing normal {en}"

