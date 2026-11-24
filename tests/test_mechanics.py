import numpy as np
import pytest

from cardiac_rl.config import load_config
from cardiac_rl.mechanics import SlabMechanics


@pytest.fixture
def model():
    return SlabMechanics(load_config("configs/smoke.yaml")["mechanics"])


def test_reference_solves_traction_and_preserves_volume(model):
    for z in ([0., 0.], [.35, .8], [1., 1.]):
        stretches = model.solve_stretches(z)
        for stretch, load in zip(stretches, model.loads):
            np.testing.assert_allclose(model.stress(stretch, model.decode(z)), load, atol=1e-7)
            l3 = 1/np.prod(stretch)
            assert np.prod(stretch)*l3 == pytest.approx(1)


def test_stress_is_energy_derivative(model):
    theta = model.decode([.4, .6])
    a, af = theta
    b, bf = model.config["b"], model.config["bf"]

    def energy(l):
        l3 = 1/np.prod(l)
        q = np.sum(l*l)+l3*l3-3
        fibre = max(l[0]*l[0]-1, 0)
        return a/(2*b)*np.expm1(b*q)+af/(2*bf)*np.expm1(bf*fibre*fibre)

    stretch = np.array([1.04, 1.09])
    gradient = np.array([(energy(stretch+np.eye(2)[i]*1e-6)-energy(stretch-np.eye(2)[i]*1e-6))/2e-6 for i in range(2)])
    np.testing.assert_allclose(model.stress(stretch, theta), gradient, rtol=1e-6)


def test_parameters_identifiable_with_known_biaxial_loads(model):
    z, h = np.array([.4, .6]), 1e-4
    jac = np.column_stack([(model.reference(z+np.eye(2)[i]*h)-model.reference(z-np.eye(2)[i]*h))/(2*h) for i in range(2)])
    singular = np.linalg.svd(jac, compute_uv=False)
    assert singular[-1] > .01
    assert singular[0]/singular[-1] < 100


def test_outside_parameter_domain_rejected(model):
    with pytest.raises(ValueError):
        model.reference([1.01, .5])
