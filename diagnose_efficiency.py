"""Summarize covariance and full-label oracle efficiency for each score source."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    moments = pd.read_csv(args.analysis/"measurement.csv")
    estimates = pd.read_csv(args.analysis/"summary.csv")
    run = json.loads((args.analysis/"run.json").read_text())
    sizes = {r["language"]:r["N"] for r in run["source_inputs"]}
    assert all(n%2==0 for n in sizes.values())
    rows,coefficients = [],[]
    for (target,model),group in moments.groupby(["pt_field","model"]):
        assert set(group.language)=={"en","pt"}
        for r in group.itertuples():
            correlation = r.z_q_covariance/np.sqrt(r.z_variance*r.q_variance) if r.z_variance*r.q_variance>0 else 0.
            raw = r.z_q_covariance/r.q_variance if r.q_variance>0 else 0.
            standardized = r.z_q_covariance/np.sqrt(r.q_variance) if r.q_variance>0 else 0.
            coefficients.append({"pt_field":target,"model":model,"language":r.language,
                "correlation":correlation,"unrestricted_oracle_raw_coefficient":raw,
                "unrestricted_oracle_standardized_coefficient":standardized,
                "restricted_oracle_standardized_coefficient":float(np.clip(standardized,0,2))})
        for n in (25,50,100):
            variance = {k:0. for k in ("direct","unrestricted","nonnegative")}
            for r in group.itertuples():
                multiplier = .25*(1/(n//2)+1/(n-n//2))-1/sizes[r.language]
                optimum = r.z_q_covariance/r.q_variance if r.q_variance>0 else 0.
                nonnegative = np.clip(r.z_q_covariance/np.sqrt(r.q_variance),0,2)/np.sqrt(r.q_variance) if r.q_variance>0 else 0.
                for name,b in [("direct",0.),("unrestricted",optimum),("nonnegative",nonnegative)]:
                    variance[name] += multiplier*max(0.,r.z_variance+b*b*r.q_variance-2*b*r.z_q_covariance)
            fitted = estimates[(estimates.pt_field==target)&(estimates.model==model)&(estimates.budget==n)
                               &(estimates.method=="crossfit_standardized")]
            assert len(fitted)==1
            rows.append({"pt_field":target,"model":model,"budget":n,
                **{k+"_oracle_constant_rmse":float(np.sqrt(v)) for k,v in variance.items()},
                "unrestricted_oracle_rmse_ratio":float(np.sqrt(variance["unrestricted"]/variance["direct"])),
                "nonnegative_oracle_rmse_ratio":float(np.sqrt(variance["nonnegative"]/variance["direct"])),
                "observed_fitted_coefficient_mc_rmse":float(fitted.iloc[0].rmse),
                "fitted_mc_mse_minus_nonnegative_oracle_mse":float(fitted.iloc[0].rmse**2-variance["nonnegative"]),
                "oracle_uses_all_corpus_human_labels":True})
    args.output.mkdir(parents=True)
    pd.DataFrame(rows).to_csv(args.output/"oracle_constant_diagnostics.csv",index=False)
    pd.DataFrame(coefficients).to_csv(args.output/"correlations_and_coefficients.csv",index=False)
    (args.output/"run.json").write_text(json.dumps({"source_analysis":str(args.analysis),"source_run":run,
        "scope":"full-label moment oracle; its labels are unavailable to the budgeted estimators",
        "oracle_class":"one constant coefficient per language; not a bound over arbitrary estimators",
        "design_variance":"equal random strata; samples floor(n/2), ceil(n/2); known stratum auxiliary means",
        "complete":True},indent=2)+"\n")
    print(json.dumps({"status":"complete","diagnostic_rows":len(rows),"output":str(args.output)}),flush=True)


if __name__=="__main__":
    main()
