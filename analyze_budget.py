"""Compare label-budget estimators with paired finite-corpus samples."""

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logsumexp
from scipy.stats import norm

HERE = Path(__file__).resolve().parent
from frame_data import load_release, PERSPECTIVES, MODELS, contrast
from budget_estimators import two_stage_estimate, rmse_comparison


def read_probabilities(path,data,scoring_directory=None):
    rows = [json.loads(line) for line in path.open() if line.strip()]
    scores = {r["id"]:r for r in rows}
    assert len(scores)==len(rows)
    expected = set()
    probabilities,diagnostics = {},[]
    for language,frame in data.items():
        p = np.empty((len(frame),15))
        for index in range(len(frame)):
            orders,mass = [],[]
            for order in (0,1):
                key = f"{language}:{index}:o{order}"
                expected.add(key)
                record = scores[key]
                assert record["labels"]==list("ABCDEFGHIJKLMNO")
                logp = np.asarray(record["answer_log_probabilities"],dtype=float)
                assert len(logp)==15 and np.isfinite(logp).all()
                q = np.exp(logp-logsumexp(logp))
                orders.append(q if order==0 else q[::-1])
                mass.append(record["answer_probability_mass"])
            p[index] = .5*(orders[0]+orders[1])
            diagnostics.append({"language":language,"source_index":index,
                "order_total_variation":float(np.abs(orders[0]-orders[1]).sum()/2),
                "order_top_disagreement":int(orders[0].argmax()!=orders[1].argmax()),
                "answer_probability_mass":float(np.mean(mass))})
        assert np.allclose(p.sum(axis=1),1)
        probabilities[language] = p
    assert set(scores)==expected
    scoring_directory = HERE/"work/inputs/policy_compact" if scoring_directory is None else scoring_directory
    items = [json.loads(line) for line in (scoring_directory/"items.jsonl").open()]
    assert len(items)==sum(map(len,data.values()))
    for row in items:
        assert str(data[row["language"]].iloc[row["source_index"]].prompt_id)==row["prompt_id"]
    return probabilities,pd.DataFrame(diagnostics)


def make_design(q,repetitions,rng):
    """Random equal strata, with nested independent within-stratum samples."""
    size,models = q.shape
    first = size//2
    samples = np.empty((repetitions,2,50),dtype=np.int32)
    means = np.empty((repetitions,2,models))
    for repeat in range(repetitions):
        order = rng.permutation(size)
        samples[repeat,0] = order[:50]
        samples[repeat,1] = order[first:first+50]
        means[repeat,0] = q[order[:first]].mean(axis=0)
        means[repeat,1] = q[order[first:]].mean(axis=0)
    return {"samples":samples,"means":means,"stratum_sizes":np.array([first,size-first])}


def slope(z,q):
    m = z.shape[1]
    covariance = ((z-z.mean(1,keepdims=True))*(q-q.mean(1,keepdims=True))).sum(1)/(m-1)
    variance = q.var(1,ddof=1)
    return np.where(variance>0,np.clip(covariance/(variance+1/m),0,2),0.)


def estimates(z,q,design,column,budget):
    n0,n1 = budget//2,budget-budget//2
    indices = [design["samples"][:,0,:n0],design["samples"][:,1,:n1]]
    union = np.concatenate(indices,axis=1)
    zs,qs = z[union],q[union]
    total = len(z)
    fpc = (1-budget/total)/budget
    values = {"direct":zs.mean(1),"fixed_one":q.mean()+(zs-qs).mean(1)}
    variances = {"direct":fpc*zs.var(1,ddof=1),"fixed_one":fpc*(zs-qs).var(1,ddof=1)}
    m = budget//5
    values["pilot_fifth"],pilot_beta = two_stage_estimate(zs[:,:m],zs[:,m:],qs[:,:m],qs[:,m:],
                                                        total,q.sum(),q.var(ddof=1))
    residual = zs[:,m:]-pilot_beta[:,None]*qs[:,m:]
    variances["pilot_fifth"] = ((total-m)/total)**2*(1-(budget-m)/(total-m))*residual.var(1,ddof=1)/(budget-m)
    beta = [slope(z[indices[1]],q[indices[1]]),slope(z[indices[0]],q[indices[0]])]
    fitted,plugin = [],[]
    for fold,(index,n) in enumerate(zip(indices,(n0,n1))):
        residual = z[index]-beta[fold][:,None]*q[index]
        population = design["stratum_sizes"][fold]
        weight = population/total
        fitted.append(weight*(beta[fold]*design["means"][:,fold,column]+residual.mean(1)))
        plugin.append(weight**2*(1-n/population)*residual.var(1,ddof=1)/n)
    values["crossfit"] = fitted[0]+fitted[1]
    variances["crossfit"] = plugin[0]+plugin[1]
    # Full auxiliary moments are known without buying any human labels.
    q_scale = q.std(ddof=1)
    normalized = (q-q.mean())/q_scale if q_scale>0 else np.zeros_like(q)
    normalized_means = (design["means"][:,:,column]-q.mean())/q_scale if q_scale>0 else np.zeros(design["means"].shape[:2])
    standardized_beta = [slope(z[indices[1]],normalized[indices[1]]),slope(z[indices[0]],normalized[indices[0]])]
    fitted,plugin = [],[]
    for fold,(index,n) in enumerate(zip(indices,(n0,n1))):
        residual = z[index]-standardized_beta[fold][:,None]*normalized[index]
        population = design["stratum_sizes"][fold]
        weight = population/total
        fitted.append(weight*(standardized_beta[fold]*normalized_means[:,fold]+residual.mean(1)))
        plugin.append(weight**2*(1-n/population)*residual.var(1,ddof=1)/n)
    values["crossfit_standardized"] = fitted[0]+fitted[1]
    variances["crossfit_standardized"] = plugin[0]+plugin[1]
    exact = {"direct":fpc*z.var(ddof=1),"fixed_one":fpc*(z-q).var(ddof=1)}
    return values,variances,exact,{"pilot_fifth":pilot_beta,"crossfit":np.concatenate(beta),
                                  "crossfit_standardized":np.concatenate(standardized_beta)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores",type=Path,required=True)
    parser.add_argument("--model-name",required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--repetitions",type=int,default=5000)
    parser.add_argument("--scoring-directory",type=Path,default=HERE/"work/inputs/policy_compact")
    args = parser.parse_args()
    assert args.repetitions>=5000 and not args.output.exists()
    data,sources = load_release(HERE/"data/framenews/data")
    probabilities,diagnostics = read_probabilities(args.scores,data,args.scoring_directory)
    names = ["llm_soft","llm_hard",*MODELS]
    category_predictions = {language:{"llm_soft":p,"llm_hard":np.eye(15)[p.argmax(1)],
          **{name:np.eye(15)[frame[name].to_numpy(dtype=int)-1] for name in MODELS}}
          for (language,frame),p in zip(data.items(),[probabilities[l] for l in data])}
    q = {language:np.stack([v[:,0]-v[:,3] for v in preds.values()],axis=1)
         for language,preds in category_predictions.items()}
    rng = np.random.default_rng(20261009)
    designs = {language:make_design(q[language],args.repetitions,rng) for language in data}
    summaries,comparisons,coefficients,measurement,pairs,vectors = [],[],[],[],[],[]
    errors_to_save = {}
    for perspective in PERSPECTIVES:
        gold = {language:np.eye(15)[frame["primary_code" if language=="en" else perspective].to_numpy(dtype=int)-1]
                for language,frame in data.items()}
        z = {language:values[:,0]-values[:,3] for language,values in gold.items()}
        truth = z["en"].mean()-z["pt"].mean()
        for language,frame in data.items():
            labels = gold[language].argmax(1)
            for name,p in category_predictions[language].items():
                covariance = np.cov(z[language],p[:,0]-p[:,3],ddof=1)[0,1]
                measurement.append({"pt_field":perspective,"language":language,"model":name,
                    "accuracy":float((p.argmax(1)==labels).mean()),
                    "brier":float(((p-gold[language])**2).sum(1).mean()),
                    "gold_nll":float(-np.log(np.maximum(p[np.arange(len(p)),labels],1e-12)).mean()),
                    "nll_clipping_floor":1e-12,"z_variance":float(z[language].var(ddof=1)),
                    "q_variance":float((p[:,0]-p[:,3]).var(ddof=1)),"z_q_covariance":float(covariance),
                    "raw_contrast_error":float((p[:,0]-p[:,3]).mean()-z[language].mean())})
        for budget in (25,50,100):
            errors = {}
            for column,name in enumerate(names):
                results = {language:estimates(z[language],q[language][:,column],designs[language],column,budget) for language in data}
                for method in ("direct","fixed_one","pilot_fifth","crossfit","crossfit_standardized"):
                    error = results["en"][0][method]-results["pt"][0][method]-truth
                    variance = results["en"][1][method]+results["pt"][1][method]
                    exact = sum(r[2][method] for r in results.values()) if method in ("direct","fixed_one") else None
                    errors[(name,method)] = error
                    errors_to_save[f"{perspective}_{budget}_{name}_{method}"] = error
                    squared = error**2
                    rmse = float(np.sqrt(squared.mean()))
                    covered = np.abs(error)<=norm.ppf(.975)*np.sqrt(variance)
                    summaries.append({"pt_field":perspective,"model":name,"method":method,"budget":budget,
                        "truth":float(truth),"bias":float(error.mean()),"bias_mcse":float(error.std(ddof=1)/np.sqrt(len(error))),
                        "mae":float(np.abs(error).mean()),"rmse":rmse,
                        "rmse_mcse":float(squared.std(ddof=1)/(2*rmse*np.sqrt(len(error)))) if rmse else 0.,
                        "exact_rmse":float(np.sqrt(exact)) if exact is not None else None,
                        "normal_interval_coverage":float(covered.mean()),
                        "coverage_mcse":float(np.sqrt(covered.mean()*(1-covered.mean())/len(error))),
                        "crossfit_interval_ignores_nuisance_cross_covariance":method.startswith("crossfit")})
                for language,result in results.items():
                    for method,beta in result[3].items():
                        coefficients.append({"pt_field":perspective,"language":language,"model":name,"method":method,
                            "budget":budget,"mean":float(beta.mean()),"fraction_zero":float((beta==0).mean()),
                            "fraction_upper_bound":float((beta==(2 if method.startswith("crossfit") else 1)).mean()),
                            **{f"q{int(v*100)}":float(np.quantile(beta,v)) for v in (.05,.5,.95)}})
            for method in ("fixed_one","pilot_fifth","crossfit","crossfit_standardized"):
                for baseline in ("llm_hard",*MODELS):
                    result = rmse_comparison(errors[("llm_soft",method)],errors[(baseline,method)])
                    squared_difference = errors[("llm_soft",method)]**2-errors[(baseline,method)]**2
                    result["paired_mse_difference"] = float(squared_difference.mean())
                    result["paired_mcse_mse_difference"] = float(squared_difference.std(ddof=1)/np.sqrt(len(squared_difference)))
                    critical = norm.ppf(1-.05/(2*6))
                    comparisons.append({"pt_field":perspective,"budget":budget,"method":method,"new":"llm_soft",
                        "baseline":baseline,**result,
                        "primary_family":method=="crossfit_standardized" and budget==50,
                        "simultaneous_mc_low":result["rmse_difference"]-critical*result["paired_mcse_rmse_difference"],
                        "simultaneous_mc_high":result["rmse_difference"]+critical*result["paired_mcse_rmse_difference"]})
                for name in names:
                    comparisons.append({"pt_field":perspective,"budget":budget,"method":method,"new":name,
                        "baseline":"direct",**rmse_comparison(errors[(name,method)],errors[(name,"direct")]),"primary_family":False})
            for name in names:
                variances = [sum((1-budget/len(data[l]))/budget*(gold[l][:,a]-gold[l][:,b]).var(ddof=1) for l in data)
                             for a,b in itertools.combinations(range(15),2)]
                for (a,b),direct in zip(itertools.combinations(range(15),2),variances):
                    residual = sum((1-budget/len(data[l]))/budget*((gold[l][:,a]-gold[l][:,b])-
                        (category_predictions[l][name][:,a]-category_predictions[l][name][:,b])).var(ddof=1) for l in data)
                    pairs.append({"pt_field":perspective,"budget":budget,"model":name,"category_a":a+1,"category_b":b+1,
                                  "fixed_one_rmse":float(np.sqrt(residual)),"direct_rmse":float(np.sqrt(direct)),
                                  "ratio":float(np.sqrt(residual/direct)) if direct else None})
                direct = sum((1-budget/len(data[l]))/budget*gold[l].var(0,ddof=1).sum() for l in data)
                residual = sum((1-budget/len(data[l]))/budget*(gold[l]-category_predictions[l][name]).var(0,ddof=1).sum() for l in data)
                vectors.append({"pt_field":perspective,"budget":budget,"model":name,"direct_total_mse":float(direct),
                                "fixed_one_total_mse":float(residual),"root_total_mse_ratio":float(np.sqrt(residual/direct))})
    args.output.mkdir(parents=True)
    for name,rows in [("summary",summaries),("comparisons",comparisons),("coefficients",coefficients),
                      ("measurement",measurement),("all_105_contrasts",pairs),("all_15_prevalences",vectors)]:
        pd.DataFrame(rows).to_csv(args.output/f"{name}.csv",index=False)
    diagnostics.to_csv(args.output/"answer_order_diagnostics.csv",index=False)
    np.savez_compressed(args.output/"private_mc_errors.npz",**errors_to_save)
    np.savez_compressed(args.output/"private_sampling_indices.npz",**{l:d["samples"] for l,d in designs.items()})
    (args.output/"run.json").write_text(json.dumps({"model":args.model_name,"source_scores":str(args.scores),
        "source_inputs":sources,"scoring_directory":str(args.scoring_directory),"seed":20261009,"repetitions":args.repetitions,"status":"complete",
        "primary":"crossfit with unlabeled full-auxiliary standardization; soft versus same-estimator hard controls, n=50, both PT targets",
        "pre_evaluation_revision":"standardized slope fitting is invariant to positive affine transforms of the auxiliary signal; raw-q crossfit remains sensitivity",
        "scope":"fixed released corpora; uncertainty on numerical integration only",
        "crossfit_intervals":"plug-in normal diagnostic; ignores covariance from reciprocal nuisance fitting",
        "joint_design":"uniform random near-equal strata, independent within-stratum samples; marginal ordered union SRS",
        "prediction_calibration":"none; all human labels used only as finite evaluation targets or counted sample labels"},indent=2)+"\n")
    print(json.dumps({"status":"complete","output":str(args.output),"repetitions":args.repetitions}),flush=True)


if __name__=="__main__":
    main()
