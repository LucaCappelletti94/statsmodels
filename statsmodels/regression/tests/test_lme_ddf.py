"""
Tests of the Satterthwaite degrees of freedom of MixedLM against lmerTest

The reference values in LMERTEST come from R 4.6.1 with lme4 2.1.0,
lmerTest 3.2.1 and pbkrtest 0.5.5, using the R script below.
"""

# library(lmerTest)
# data(beets, package = "pbkrtest")
# beets$bh <- interaction(beets$block, beets$harvest)
# ub <- beets[-c(1, 7, 14, 22), ]
# ss <- sleepstudy[sleepstudy$Days < 10 - (as.integer(sleepstudy$Subject) %% 4), ]
# ss$Days2 <- (ss$Days - 4.5)^2 / 10
# models <- list(
#   beets_unbalanced = lmer(sugpct ~ block + sow + harvest + (1 | bh), data = ub),
#   beets_balanced = lmer(sugpct ~ block + sow + harvest + (1 | bh), data = beets),
#   sleep_balanced = lmer(Reaction ~ Days + (Days | Subject), data = sleepstudy),
#   sleep_unbalanced = lmer(Reaction ~ Days + (Days | Subject), data = ss),
#   sleep_quadratic = lmer(Reaction ~ Days + Days2 + (Days + Days2 | Subject), data = ss),
#   beets_vc = lmer(sugpct ~ sow + harvest + (1 | block) + (1 | block:harvest), data = ub),
#   pastes_crossed = lmer(strength ~ 1 + (1 | batch) + (1 | sample), data = Pastes),
#   dyestuff2 = lmer(Yield ~ 1 + (1 | Batch), data = Dyestuff2)
# )
# vec <- function(x) paste0("[", paste(sprintf("%.12g", x), collapse = ", "), "]")
# set.seed(1)
# for (name in names(models)) {
#   m <- models[[name]]
#   cf <- coef(summary(m, ddf = "Satterthwaite"))
#   v <- as.matrix(vcov(m))
#   p <- nrow(v)
#   L <- matrix(round(rnorm(2 * p), 2), 2, p)
#   ct <- t(apply(L, 1, function(l) unlist(contest1D(m, l)[c("Std. Error", "df")])))
#   cat(sprintf('    "%s": {\n', name))
#   cat(sprintf('        "theta": %s,\n', vec(getME(m, "theta"))))
#   cat('        "vcov": [\n')
#   for (i in seq_len(p)) cat(sprintf("            %s,\n", vec(v[i, ])))
#   cat("        ],\n")
#   cat(sprintf('        "se": %s,\n', vec(cf[, "Std. Error"])))
#   cat(sprintf('        "df": %s,\n', vec(cf[, "df"])))
#   cat(sprintf('        "t": %s,\n', vec(cf[, "t value"])))
#   cat(sprintf('        "p": %s,\n', vec(cf[, "Pr(>|t|)"])))
#   cat(sprintf('        "L": [%s, %s],\n', vec(L[1, ]), vec(L[2, ])))
#   cat(sprintf('        "L_se": %s,\n', vec(ct[, 1])))
#   cat(sprintf('        "L_df": %s,\n', vec(ct[, 2])))
#   cat("    },\n")
# }

from pathlib import Path
import warnings

import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
import pytest

from statsmodels.base import _penalties as penalties
from statsmodels.regression.mixed_linear_model import (
    MixedLM,
    MixedLMParams,
    _pinv_lmertest,
)
from statsmodels.tools.sm_exceptions import ConvergenceWarning, SingularMatrixWarning

# The beets data of the R package pbkrtest, a split-plot sugar beet trial.
# The columns are harvest, block, sow and sugpct.
BEETS = pd.DataFrame(
    [
        ("harv1", "block1", "sow3", 17.1),
        ("harv1", "block1", "sow4", 16.9),
        ("harv1", "block1", "sow5", 16.6),
        ("harv1", "block1", "sow2", 17.0),
        ("harv1", "block1", "sow1", 17.0),
        ("harv2", "block2", "sow3", 17.0),
        ("harv2", "block2", "sow2", 17.0),
        ("harv2", "block2", "sow4", 16.7),
        ("harv2", "block2", "sow5", 16.4),
        ("harv2", "block2", "sow1", 16.8),
        ("harv1", "block3", "sow5", 16.6),
        ("harv1", "block3", "sow2", 16.9),
        ("harv1", "block3", "sow3", 17.1),
        ("harv1", "block3", "sow4", 16.8),
        ("harv1", "block3", "sow1", 16.9),
        ("harv2", "block1", "sow2", 17.0),
        ("harv2", "block1", "sow1", 16.8),
        ("harv2", "block1", "sow5", 16.5),
        ("harv2", "block1", "sow4", 16.7),
        ("harv2", "block1", "sow3", 17.0),
        ("harv1", "block2", "sow4", 16.8),
        ("harv1", "block2", "sow1", 16.9),
        ("harv1", "block2", "sow3", 17.0),
        ("harv1", "block2", "sow2", 17.0),
        ("harv1", "block2", "sow5", 16.5),
        ("harv2", "block3", "sow1", 16.7),
        ("harv2", "block3", "sow4", 16.6),
        ("harv2", "block3", "sow3", 16.9),
        ("harv2", "block3", "sow2", 16.9),
        ("harv2", "block3", "sow5", 16.4),
    ],
    columns=["harvest", "block", "sow", "sugpct"],
)
BEETS["bh"] = BEETS["block"] + "." + BEETS["harvest"]
# The rows that the R script drops, 1, 7, 14 and 22 counting from one
BEETS_UNBALANCED = BEETS.drop(index=[0, 6, 13, 21])

# The Dyestuff2 data of the R package lme4
DYESTUFF2 = pd.DataFrame(
    {
        "Batch": np.repeat(list("ABCDEF"), 5),
        "Yield": [
            7.298,
            3.846,
            2.434,
            9.566,
            7.99,
            5.22,
            6.556,
            0.608,
            11.788,
            -0.892,
            0.11,
            10.386,
            13.434,
            5.51,
            8.166,
            2.212,
            4.852,
            7.092,
            9.288,
            4.98,
            0.282,
            9.014,
            4.458,
            9.446,
            7.198,
            1.722,
            4.782,
            8.106,
            0.758,
            3.758,
        ],
    }
)

# The sleepstudy data of the R package lme4
SLEEPSTUDY = pd.read_csv(Path(__file__).parent / "results" / "sleepstudy.csv")
# as.integer(Subject) in R, the rank of the subject among the sorted levels
_subject_ix = SLEEPSTUDY["Subject"].rank(method="dense").astype(int)
SLEEPSTUDY_UNBALANCED = SLEEPSTUDY[SLEEPSTUDY["Days"] < 10 - _subject_ix % 4].copy()
SLEEPSTUDY_UNBALANCED["Days2"] = (SLEEPSTUDY_UNBALANCED["Days"] - 4.5) ** 2 / 10

# The Pastes data of the R package lme4
PASTES = pd.read_csv(Path(__file__).parent / "results" / "pastes.csv")

LMERTEST = {
    "beets_unbalanced": {
        "theta": [0.326741126364],
        "vcov": [
            [
                0.00101945935518,
                -0.000397379166996,
                -0.000429972462595,
                -0.000569554346162,
                -0.000464051512402,
                -0.000525227639296,
                -0.000525983035637,
                -0.00034414069616,
            ],
            [
                -0.000397379166996,
                0.000903634690048,
                0.000429972462595,
                4.30718711466e-06,
                -0.000128568397452,
                -6.73922705575e-05,
                -6.22929439834e-05,
                3.02727868628e-05,
            ],
            [
                -0.000429972462595,
                0.000429972462595,
                0.000859944925191,
                3.09124858793e-19,
                -6.1176126894e-05,
                6.1176126894e-05,
                2.64656585319e-19,
                1.01153812446e-19,
            ],
            [
                -0.000569554346162,
                4.30718711466e-06,
                3.09124858793e-19,
                0.00105316364533,
                0.000519977469089,
                0.000519977469089,
                0.000526007531049,
                8.42221721498e-05,
            ],
            [
                -0.000464051512402,
                -0.000128568397452,
                -6.1176126894e-05,
                0.000519977469089,
                0.00105947318964,
                0.000530178808715,
                0.000529453280741,
                -4.30718711466e-06,
            ],
            [
                -0.000525227639296,
                -6.73922705575e-05,
                6.1176126894e-05,
                0.000519977469089,
                0.000530178808715,
                0.00105947318964,
                0.000529453280741,
                -4.30718711466e-06,
            ],
            [
                -0.000525983035637,
                -6.22929439834e-05,
                2.64656585319e-19,
                0.000526007531049,
                0.000529453280741,
                0.000529453280741,
                0.000955044162489,
                3.80747144932e-05,
            ],
            [
                -0.00034414069616,
                3.02727868628e-05,
                1.01153812446e-19,
                8.42221721498e-05,
                -4.30718711466e-06,
                -4.30718711466e-06,
                3.80747144932e-05,
                0.000591950105425,
            ],
        ],
        "se": [
            0.0319289736005,
            0.0300605171287,
            0.0293248175645,
            0.0324524828839,
            0.032549549761,
            0.032549549761,
            0.0309037888048,
            0.0243300247724,
        ],
        "df": [
            5.17809252703,
            1.95294506485,
            1.74709445239,
            16.6676549461,
            16.5035941529,
            16.5035941544,
            16.151644503,
            1.88871757432,
        ],
        "t": [
            531.157456468,
            -2.24466048774,
            -2.92190195204,
            3.03379244712,
            5.37852813707,
            -3.10516533445,
            -11.24420317,
            -4.99552523214,
        ],
        "p": [
            1.82622771004e-13,
            0.156966571268,
            0.11671895643,
            0.00762327514741,
            5.5335538231e-05,
            0.00661115221288,
            4.73878209797e-09,
            0.0424214190135,
        ],
        "L": [
            [-0.63, -0.84, 0.33, 0.49, 0.58, 1.51, -0.62, 1.12],
            [0.18, 1.6, -0.82, 0.74, -0.31, 0.39, -2.21, -0.04],
        ],
        "L_se": [0.0841718364753, 0.0767904558505],
        "L_df": [14.9944707381, 13.9598931667],
    },
    "beets_balanced": {
        "theta": [0.553774924408],
        "vcov": [
            [
                0.00117777777812,
                -0.000633333333602,
                -0.000633333333602,
                -0.000416666666649,
                -0.000416666666649,
                -0.000416666666649,
                -0.000416666666649,
                -0.000422222222401,
            ],
            [
                -0.000633333333602,
                0.0012666666672,
                0.000633333333602,
                2.05082864358e-19,
                2.05082864358e-19,
                2.05082864358e-19,
                2.05082864358e-19,
                5.70013172973e-19,
            ],
            [
                -0.000633333333602,
                0.000633333333602,
                0.0012666666672,
                2.34380416409e-19,
                2.34380416409e-19,
                2.34380416409e-19,
                2.34380416409e-19,
                5.70013172973e-19,
            ],
            [
                -0.000416666666649,
                2.05082864358e-19,
                2.34380416409e-19,
                0.000833333333298,
                0.000416666666649,
                0.000416666666649,
                0.000416666666649,
                3.59383305161e-19,
            ],
            [
                -0.000416666666649,
                2.05082864358e-19,
                2.34380416409e-19,
                0.000416666666649,
                0.000833333333298,
                0.000416666666649,
                0.000416666666649,
                3.75008666255e-19,
            ],
            [
                -0.000416666666649,
                2.05082864358e-19,
                2.34380416409e-19,
                0.000416666666649,
                0.000416666666649,
                0.000833333333298,
                0.000416666666649,
                3.75008666255e-19,
            ],
            [
                -0.000416666666649,
                2.05082864358e-19,
                2.34380416409e-19,
                0.000416666666649,
                0.000416666666649,
                0.000416666666649,
                0.000833333333298,
                3.75008666255e-19,
            ],
            [
                -0.000422222222401,
                5.70013172973e-19,
                5.70013172973e-19,
                3.59383305161e-19,
                3.75008666255e-19,
                3.75008666255e-19,
                3.75008666255e-19,
                0.000844444444803,
            ],
        ],
        "se": [
            0.0343187671416,
            0.0355902608477,
            0.0355902608477,
            0.0288675134589,
            0.0288675134589,
            0.0288675134589,
            0.0288675134589,
            0.0290593262964,
        ],
        "df": [
            3.83088987224,
            1.99999999928,
            1.99999999926,
            19.9999999998,
            20.0000000006,
            19.9999999996,
            19.9999999999,
            1.99999999918,
        ],
        "t": [
            493.898860936,
            -1.40487871707,
            -2.24780594732,
            4.04145188441,
            5.77350269202,
            -3.46410161521,
            -12.1243556532,
            -3.90006747497,
        ],
        "p": [
            2.40151427376e-10,
            0.295238521515,
            0.153585261067,
            0.000638531951926,
            1.19449112022e-05,
            0.00245042816042,
            1.13288050546e-10,
            0.0598978519268,
        ],
        "L": [
            [-0.02, 0.82, 0.92, 0.07, 0.62, -0.16, -0.48, 1.36],
            [0.94, 0.59, 0.78, -1.99, -0.06, -1.47, 0.42, -0.1],
        ],
        "L_se": [0.0691342253375, 0.101733458934],
        "L_df": [2.2467311965, 21.9921074319],
    },
    "sleep_balanced": {
        "theta": [0.966741773967, 0.015169058896, 0.230909953208],
        "vcov": [
            [46.5751200488, -1.45108841663],
            [-1.45108841663, 2.38946562321],
        ],
        "se": [6.82459669495, 1.54578964391],
        "df": [16.9997262885, 16.9999837256],
        "t": [36.8380896463, 6.771481489],
        "p": [1.17155811121e-17, 3.26382446993e-06],
        "L": [[0.39, -1.38], [-0.05, -0.41]],
        "L_se": [3.63270225368, 0.677209233768],
        "L_df": [16.9998388097, 16.9998697983],
    },
    "sleep_unbalanced": {
        "theta": [1.08678121277, -0.0216282661413, 0.298151406324],
        "vcov": [
            [48.7612889486, -3.07525046869],
            [-3.07525046869, 3.4150348471],
        ],
        "se": [6.98292839349, 1.84798128971],
        "df": [16.9480578171, 17.6050373756],
        "t": [36.1096454388, 5.47915760507],
        "p": [1.79005658817e-17, 3.60099282309e-05],
        "L": [[-0.39, 1.1], [-0.06, 0.76]],
        "L_se": [3.76660976426, 1.55837338615],
        "L_df": [18.0371647725, 18.2564966083],
    },
    "sleep_quadratic": {
        "theta": [
            1.39020882964,
            -0.0372661027254,
            -0.266372021948,
            0.300900497148,
            -0.0340288706308,
            0.383175945739,
        ],
        "vcov": [
            [82.6673861466, -5.87955709865, -24.1481811608],
            [-5.87955709865, 3.5166666483, 1.78660198717],
            [-24.1481811608, 1.78660198717, 17.4270107925],
        ],
        "se": [9.09216069736, 1.87527775231, 4.17456713834],
        "df": [15.192856896, 17.0501986856, 14.714350173],
        "t": [27.3534257895, 5.55009504354, 0.808223460404],
        "p": [2.42906842254e-14, 3.48855316151e-05, 0.431835437294],
        "L": [[-0.16, 0.7, -0.69], [-0.25, 0.56, -0.71]],
        "L_se": [2.52897142165, 2.58987648565],
        "L_df": [17.4485286337, 17.2490510637],
    },
    "beets_vc": {
        "theta": [0.341688943987, 0.792365084837],
        "vcov": [
            [
                0.00129924663842,
                -0.00056651685729,
                -0.000524196450833,
                -0.000524196450833,
                -0.000544007197525,
                -0.000341867562149,
            ],
            [
                -0.00056651685729,
                0.0010499512976,
                0.000518708216237,
                0.000518708216237,
                0.000524505176577,
                8.40233614259e-05,
            ],
            [
                -0.000524196450833,
                0.000518708216237,
                0.00105209862811,
                0.000526301796317,
                0.000525810637479,
                -3.22837329161e-06,
            ],
            [
                -0.000524196450833,
                0.000518708216237,
                0.000526301796317,
                0.00105209862811,
                0.000525810637479,
                -3.22837329161e-06,
            ],
            [
                -0.000544007197525,
                0.000524505176577,
                0.000525810637479,
                0.000525810637479,
                0.000950442763553,
                3.86156826608e-05,
            ],
            [
                -0.000341867562149,
                8.40233614259e-05,
                -3.22837329161e-06,
                -3.22837329161e-06,
                3.86156826608e-05,
                0.000606503758975,
            ],
        ],
        "se": [
            0.0360450639952,
            0.032402951989,
            0.0324360698622,
            0.0324360698622,
            0.0308292517514,
            0.0246272970294,
        ],
        "df": [
            6.53421926923,
            16.7610292969,
            16.7178021598,
            16.7178021599,
            16.3087646072,
            1.89760386195,
        ],
        "t": [
            469.093791992,
            3.0451495159,
            5.33718688537,
            -3.09509306613,
            -11.2842391593,
            -4.92389417961,
        ],
        "p": [
            4.85829301612e-16,
            0.00740547205415,
            5.75947969254e-05,
            0.00667430229256,
            4.05023817276e-09,
            0.043132287733,
        ],
        "L": [
            [0.36, -0.11, 0.4, 0.34, 1.43, -0.37],
            [0.77, 0.88, -0.61, -1.13, 1.98, -1.04],
        ],
        "L_se": [0.0526160131696, 0.0651411571664],
        "L_df": [18.8974860416, 13.9931637843],
    },
    "pastes_crossed": {
        "theta": [3.52690220897, 1.56346001184],
        "vcov": [
            [0.458153059516],
        ],
        "se": [0.676870046254],
        "df": [9.00000034828],
        "t": [88.7221020721],
        "p": [1.48783161495e-14],
        "L": [[0.57], [-0.14]],
        "L_se": [0.385815926365, 0.0947618064756],
        "L_df": [9.00000034828, 9.00000034828],
    },
    "dyestuff2": {
        "theta": [0],
        "vcov": [
            [0.46021032092],
        ],
        "se": [0.678388031233],
        "df": [28.9999999994],
        "t": [8.35156243796],
        "p": [3.3182927492e-09],
        "L": [[2.4], [-0.04]],
        "L_se": [1.62813127496, 0.0271355212493],
        "L_df": [28.9999999994, 28.9999999994],
    },
}

BEETS_FORMULA = "sugpct ~ block + sow + harvest"
VC_FORMULA = "sugpct ~ sow + harvest"


def _model(name, use_sparse=False):
    """
    Return the model and the variance parameters relative to the scale

    The variance parameters are those of lme4, given by theta.
    """
    theta = np.asarray(LMERTEST[name]["theta"])
    if name.startswith("beets_") and name != "beets_vc":
        data = BEETS_UNBALANCED if name == "beets_unbalanced" else BEETS
        model = MixedLM.from_formula(BEETS_FORMULA, data, groups=data["bh"])
        return model, np.diag(theta**2), np.empty(0)
    if name.startswith("sleep_"):
        data = SLEEPSTUDY if name == "sleep_balanced" else SLEEPSTUDY_UNBALANCED
        quadratic = name == "sleep_quadratic"
        terms = "Days + Days2" if quadratic else "Days"
        model = MixedLM.from_formula(
            f"Reaction ~ {terms}", data, groups=data["Subject"], re_formula=terms
        )
        # theta is the lower triangle of the Cholesky factor, column by column
        lam = np.zeros((model.k_re, model.k_re))
        lam.T[np.triu_indices(model.k_re)] = theta
        return model, lam @ lam.T, np.empty(0)
    if name == "beets_vc":
        # lme4 orders the terms (1 | block:harvest), (1 | block)
        data = BEETS_UNBALANCED
        model = MixedLM.from_formula(
            VC_FORMULA,
            data,
            groups=data["block"],
            re_formula="1",
            vc_formula={"bh": "0 + C(harvest)"},
            use_sparse=use_sparse,
        )
        return model, np.diag(theta[1:] ** 2), theta[:1] ** 2
    if name == "pastes_crossed":
        # A single group with crossed variance components, lme4 orders the
        # terms (1 | sample), (1 | batch)
        model = MixedLM.from_formula(
            "strength ~ 1",
            PASTES,
            groups=np.ones(len(PASTES)),
            re_formula="0",
            vc_formula={"batch": "0 + C(batch)", "sample": "0 + C(sample)"},
        )
        vcomp = {"sample": theta[0] ** 2, "batch": theta[1] ** 2}
        vcomp = np.array([vcomp[vc_name] for vc_name in model.exog_vc.names])
        return model, np.zeros((0, 0)), vcomp
    model = MixedLM.from_formula("Yield ~ 1", DYESTUFF2, groups=DYESTUFF2["Batch"])
    return model, np.diag(theta**2), np.empty(0)


def _fit_at_lme4(name, use_sparse=False):
    """Return the results of a REML fit at the lme4 estimates"""
    model, cov_re, vcomp = _model(name, use_sparse)
    start = MixedLMParams.from_components(cov_re=cov_re, vcomp=vcomp)
    with warnings.catch_warnings():
        # The Dyestuff2 estimate is on the boundary
        warnings.simplefilter("ignore", ConvergenceWarning)
        warnings.simplefilter("ignore", SingularMatrixWarning)
        return model.fit(start_params=start, do_cg=False)


@pytest.mark.parametrize(
    "name, use_sparse",
    [(name, False) for name in LMERTEST] + [("beets_vc", True)],
)
def test_satterthwaite_lmertest(name, use_sparse):
    ref = LMERTEST[name]
    res = _fit_at_lme4(name, use_sparse)
    with warnings.catch_warnings():
        # lmerTest does not warn for any of these models, including the
        # singular fit of Dyestuff2
        warnings.simplefilter("error")
        tt = res.t_test(np.eye(res.k_fe), df_method="satterthwaite")
        tt_l = res.t_test(np.asarray(ref["L"]), df_method="satterthwaite")

    vcov = np.asarray(ref["vcov"])
    assert_allclose(res._satterthwaite[0], vcov, rtol=1e-7, atol=1e-12 * vcov.max())
    assert_allclose(tt.sd, ref["se"], rtol=1e-7)
    assert_allclose(tt.tvalue, ref["t"], rtol=1e-7)
    # lmerTest differentiates numerically
    assert_allclose(tt.df_denom, ref["df"], rtol=1e-5)
    assert_allclose(tt.pvalue, ref["p"], rtol=1e-5)
    # Contrasts of several coefficients, against lmerTest::contest1D
    assert_allclose(tt_l.sd, ref["L_se"], rtol=1e-7)
    assert_allclose(tt_l.df_denom, ref["L_df"], rtol=1e-5)


def test_satterthwaite_fitted():
    # The variance parameters estimated by statsmodels differ from those of
    # lme4 in the seventh digit
    ref = LMERTEST["beets_unbalanced"]
    data = BEETS_UNBALANCED
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        res = MixedLM.from_formula(BEETS_FORMULA, data, groups=data["bh"]).fit()
    tt = res.t_test(np.eye(res.k_fe), df_method="satterthwaite")
    assert_allclose(tt.sd, ref["se"], rtol=1e-5)
    assert_allclose(tt.df_denom, ref["df"], rtol=1e-5)


@pytest.mark.parametrize(
    "evals, messages",
    [
        (
            [2.0, 1e-9, -0.5],
            [
                "Model failed to converge with 1 negative eigenvalue: -5.0e-01",
                "Model may not have converged with 1 eigenvalue close to zero: 1.0e-09",
            ],
        ),
        (
            [4.0, -0.25, -2.0],
            ["Model failed to converge with 2 negative eigenvalues: -2.5e-01 -2.0e+00"],
        ),
        (
            [3e-9, 1.5, -1e-9],
            [
                (
                    "Model may not have converged with 2 eigenvalues close to zero: "
                    "3.0e-09 -1.0e-09"
                )
            ],
        ),
    ],
)
def test_pinv_lmertest_warns(evals, messages):
    # The messages and the dropped eigenvalues follow as_lmerModLT in lmerTest
    rng = np.random.default_rng(3)
    q, _ = np.linalg.qr(rng.standard_normal((3, 3)))
    evals = np.asarray(evals)
    with pytest.warns(ConvergenceWarning) as record:
        inv = _pinv_lmertest(q @ np.diag(evals) @ q.T)
    assert [str(w.message) for w in record] == messages
    keep = evals > 1e-8
    expected = q[:, keep] @ np.diag(1 / evals[keep]) @ q[:, keep].T
    assert_allclose(inv, expected, atol=1e-12)


def _fit_beets(**kwargs):
    data = BEETS_UNBALANCED
    model = MixedLM.from_formula(BEETS_FORMULA, data, groups=data["bh"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        return model.fit(**kwargs)


@pytest.mark.parametrize(
    "fit_kwargs, test_kwargs, match",
    [
        ({"reml": False}, {"df_method": "satterthwaite"}, "REML"),
        ({"cov_pen": penalties.PSD(1)}, {"df_method": "satterthwaite"}, "cov_pen"),
        (
            {"free": MixedLMParams.from_components(np.ones(8), np.ones((1, 1)))},
            {"df_method": "satterthwaite"},
            "free",
        ),
        ({}, {"df_method": "kenward-roger"}, "df_method"),
        ({}, {"df_method": "satterthwaite", "use_t": False}, "use_t"),
    ],
)
def test_satterthwaite_invalid(fit_kwargs, test_kwargs, match):
    res = _fit_beets(**fit_kwargs)
    with pytest.raises(ValueError, match=match):
        res.t_test(np.eye(res.k_fe), **test_kwargs)
