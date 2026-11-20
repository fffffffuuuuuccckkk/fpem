import numpy as np

from models.predictive_env.mechanism_grouping import (
    K, fit_mixture, fit_rff_mixture, fit_fingerprint, fit_fingerprint_cluster,
    posterior, predict_linear, weighted_linear, fingerprint_posterior,
)


def test_conditional_mapping_changes_with_equal_x_distribution():
    rng=np.random.default_rng(19)
    x=rng.normal(size=(320,4))
    mechanism=rng.integers(2,size=len(x))
    y=np.stack((x[:,0]*(1-2*mechanism),x[:,1]*(1-2*mechanism)),axis=1)+.03*rng.normal(size=(len(x),2))
    fit=fit_mixture("G1_Switching_Ridge",x,y,seed=19,iterations=8)
    assert fit["q_train"].shape==(len(x),K)
    assert np.allclose(fit["q_train"].sum(axis=1),1,atol=1e-5)
    assert np.isfinite(fit["q_train"]).all()


def test_noise_variance_cannot_collapse_and_all_methods_finite():
    rng=np.random.default_rng(31)
    x=rng.normal(size=(120,4));y=x[:,:2]+rng.normal(scale=.1,size=(120,2))
    for method in ("G1_Switching_Ridge","G2_HRM_lite","G5_SSCM_lite","G6_CCSL_proxy"):
        fit=fit_mixture(method,x,y,seed=31,iterations=2)
        assert np.isfinite(fit["q_train"]).all()
    for method in ("G3_KerHRM_lite","G4_ANM_MM_lite"):
        fit=fit_rff_mixture(method,x,y,seed=31,iterations=2)
        assert np.min(fit["variance"])>=1e-3


def test_signed_fingerprint_not_scalar_error_bin():
    rng=np.random.default_rng(41)
    x=rng.normal(size=(64,3));y=rng.normal(size=(64,2));reference=np.zeros_like(y)
    a=fit_fingerprint(x,y,reference,signed_only=True)
    b=fit_fingerprint(x,y,reference,signed_only=False)
    assert a.shape==(64,8) and b.shape==(64,8)
    assert np.allclose(np.linalg.norm(a,axis=1),1,atol=1e-4)
    fit=fit_fingerprint_cluster(a,seed=41)
    assert fingerprint_posterior(fit,a).shape==(64,K)
