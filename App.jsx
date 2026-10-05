import React, { useEffect, useMemo, useState } from "react";
import DoseLadder from "./DoseLadder.jsx";
import { getPolicies, getRecommendation, getSimilarPatients } from "./api.js";

const FIELDS = [
  { key: "age", label: "Age", unit: "years", min: 18, max: 46, step: 1 },
  { key: "amh_ng_ml", label: "AMH", unit: "ng/mL", min: 0.05, max: 25, step: 0.1 },
  { key: "afc", label: "Antral follicle count", unit: "follicles", min: 0, max: 60, step: 1 },
  { key: "fsh_iu_l", label: "Basal FSH", unit: "IU/L", min: 0.5, max: 40, step: 0.1 },
  { key: "bmi", label: "BMI", unit: "kg/m2", min: 15, max: 50, step: 0.1 },
];

const PRESETS = {
  "High responder": { age: 31, amh_ng_ml: 4.2, afc: 19, fsh_iu_l: 5.6, bmi: 22.5, pcos: 0, first_cycle: 1, previous_oocytes: "", previous_ohss: 0 },
  "Typical": { age: 34, amh_ng_ml: 2.4, afc: 12, fsh_iu_l: 6.8, bmi: 24.5, pcos: 0, first_cycle: 1, previous_oocytes: "", previous_ohss: 0 },
  "Low reserve": { age: 41, amh_ng_ml: 0.5, afc: 4, fsh_iu_l: 12.4, bmi: 27.2, pcos: 0, first_cycle: 0, previous_oocytes: 3, previous_ohss: 0 },
};

const REGIMEN_NAMES = {
  antagonist_agonist: "Antagonist, agonist trigger",
  antagonist_hcg: "Antagonist, hCG trigger",
  long_agonist_hcg: "Long agonist, hCG trigger",
};
const words = (text) => text.replace("gnrh_agonist", "GnRH agonist").replace("hcg", "hCG").replace("long_agonist", "long agonist");
const percent = (value) => (value === null || value === undefined ? "none" : `${(100 * value).toFixed(0)}%`);

function toPayload(form) {
  const payload = { ...form };
  FIELDS.forEach((f) => (payload[f.key] = Number(form[f.key])));
  if (form.first_cycle || form.previous_oocytes === "") delete payload.previous_oocytes;
  else payload.previous_oocytes = Number(form.previous_oocytes);
  if (form.first_cycle) payload.previous_ohss = 0;
  return payload;
}

export default function App() {
  const [form, setForm] = useState(PRESETS["High responder"]);
  const [result, setResult] = useState(null);
  const [similar, setSimilar] = useState(null);
  const [policies, setPolicies] = useState([]);
  const [regimen, setRegimen] = useState(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  const assess = async (values) => {
    setBusy(true);
    setError("");
    try {
      const payload = toPayload(values);
      const [recommendation, cohort] = await Promise.all([getRecommendation(payload), getSimilarPatients(payload)]);
      setResult(recommendation);
      setSimilar(cohort);
      setRegimen(recommendation.recommendation.regimen);
    } catch (problem) {
      setError(`The assessment did not run. ${problem.message}`);
    } finally {
      setBusy(false);
    }
  };

  useEffect(() => {
    assess(form);
    getPolicies().then((body) => setPolicies(body.policies)).catch(() => setPolicies([]));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const update = (key, value) => setForm((current) => ({ ...current, [key]: value }));
  const ladder = useMemo(() => (result ? result.options.filter((o) => o.regimen === regimen) : []), [result, regimen]);
  const rec = result?.recommendation;
  const usual = result?.comparators.usual_practice;

  return (
    <div className="shell">
      <aside className="intake">
        <h1>StimSafe AI</h1>
        <p className="lede">Starting dose and protocol for an ovarian stimulation cycle, checked against OHSS risk.</p>
        <div className="presets" role="group" aria-label="Example patients">
          {Object.keys(PRESETS).map((name) => (
            <button key={name} type="button" onClick={() => { setForm(PRESETS[name]); assess(PRESETS[name]); }}>{name}</button>
          ))}
        </div>
        <form onSubmit={(event) => { event.preventDefault(); assess(form); }}>
          {FIELDS.map((f) => (
            <label key={f.key}>
              <span>{f.label} <em>{f.unit}</em></span>
              <input type="number" required min={f.min} max={f.max} step={f.step} value={form[f.key]} onChange={(e) => update(f.key, e.target.value)} />
            </label>
          ))}
          <label className="check"><input type="checkbox" checked={!!form.pcos} onChange={(e) => update("pcos", e.target.checked ? 1 : 0)} /> Polycystic ovary syndrome</label>
          <label className="check"><input type="checkbox" checked={!!form.first_cycle} onChange={(e) => update("first_cycle", e.target.checked ? 1 : 0)} /> First stimulation cycle</label>
          {!form.first_cycle && (
            <>
              <label>
                <span>Oocytes in previous cycle</span>
                <input type="number" min="0" max="60" step="1" value={form.previous_oocytes} onChange={(e) => update("previous_oocytes", e.target.value)} />
              </label>
              <label className="check"><input type="checkbox" checked={!!form.previous_ohss} onChange={(e) => update("previous_ohss", e.target.checked ? 1 : 0)} /> OHSS in a previous cycle</label>
            </>
          )}
          <button className="primary" type="submit" disabled={busy}>{busy ? "Assessing" : "Assess patient"}</button>
        </form>
        <p className="fineprint">Research prototype trained on simulated cycles. Not a medical device.</p>
      </aside>

      <main>
        {error && <p className="error" role="alert">{error}</p>}
        {!result && !error && <p className="empty">Enter the baseline work up and choose Assess patient.</p>}
        {result && (
          <>
            <section className="verdict">
              <h2>
                Start at {rec.dose_iu} IU daily on the {words(rec.protocol)} protocol with {words(rec.trigger) === "hCG" ? "an hCG" : "a GnRH agonist"} trigger.
              </h2>
              <p>
                Expected yield {rec.expected_oocytes.toFixed(1)} oocytes (80% range {rec.oocytes_low.toFixed(0)} to {rec.oocytes_high.toFixed(0)}),
                predicted OHSS risk {(100 * rec.ohss_risk).toFixed(1)}%.
                {usual.arm !== rec.arm && (
                  <> Usual practice for this profile would be {usual.dose_iu} IU with {words(usual.trigger)} trigger: {usual.expected_oocytes.toFixed(1)} oocytes at {(100 * usual.ohss_risk).toFixed(1)}% OHSS risk.</>
                )}
              </p>
            </section>

            {result.safety.high_risk ? (
              <section className="safety flagged">
                <h3>High risk of OHSS: protective restrictions applied</h3>
                <div className="columns">
                  <ul>{result.safety.reasons.map((reason) => <li key={reason}>{reason}</li>)}</ul>
                  <ul>{result.safety.constraints.map((rule) => <li key={rule}>{rule}</li>)}</ul>
                </div>
                {result.safety.above_risk_ceiling && (
                  <p className="ceiling">Predicted OHSS risk stays above the ceiling on every permitted option. The lowest risk option is shown and this patient needs individual review.</p>
                )}
              </section>
            ) : (
              <section className="safety clear"><h3>No OHSS risk criteria met</h3>
                <p>All protocols stay available, subject to a predicted OHSS risk ceiling of 5%.</p></section>
            )}

            <section>
              <div className="section-head">
                <h3>Dose options</h3>
                <div className="tabs" role="tablist">
                  {Object.entries(REGIMEN_NAMES).map(([key, name]) => (
                    <button key={key} role="tab" aria-selected={regimen === key} className={regimen === key ? "active" : ""} onClick={() => setRegimen(key)}>{name}</button>
                  ))}
                </div>
              </div>
              <DoseLadder options={ladder} recommendedArm={rec.arm} riskCeiling={0.05} />
            </section>

            {similar && (
              <section>
                <h3>What happened to the {similar.cohort_size} most similar past patients</h3>
                <table>
                  <thead><tr><th>Dose given</th><th>Cycles</th><th>Mean oocytes</th><th>In target range</th><th>Poor response</th><th>OHSS</th></tr></thead>
                  <tbody>
                    {similar.by_dose.map((row) => (
                      <tr key={row.dose_iu} className={row.dose_iu === rec.dose_iu ? "match" : ""}>
                        <td>{row.dose_iu} IU</td>
                        <td>{row.cycles}</td>
                        <td>{row.mean_oocytes ?? "none"}</td>
                        <td><span className="bar good" style={{ width: `${70 * (row.optimal_response_rate ?? 0)}px` }} />{percent(row.optimal_response_rate)}</td>
                        <td><span className="bar poor" style={{ width: `${70 * (row.poor_response_rate ?? 0)}px` }} />{percent(row.poor_response_rate)}</td>
                        <td><span className="bar risk" style={{ width: `${280 * (row.ohss_rate ?? 0)}px` }} />{percent(row.ohss_rate)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
                <p className="note">Historical outcomes reflect who was given each dose, so they are context for the prediction above, not a like for like trial.</p>
              </section>
            )}

            {policies.length > 0 && (
              <section>
                <h3>How each prescribing policy performs across the test cohort</h3>
                <table>
                  <thead><tr><th>Policy</th><th>Utility per cycle</th><th>In target range</th><th>Poor response</th><th>OHSS</th><th>OHSS, high risk patients</th></tr></thead>
                  <tbody>
                    {policies.map((p) => (
                      <tr key={p.policy} className={p.policy.startsWith("StimSafe") ? "match" : ""}>
                        <td>{p.policy}</td>
                        <td>{p.true_value.toFixed(3)}</td>
                        <td>{(100 * p.true_optimal_response_rate).toFixed(1)}%</td>
                        <td>{(100 * p.true_poor_response_rate).toFixed(1)}%</td>
                        <td>{(100 * p.true_ohss_rate).toFixed(2)}%</td>
                        <td>{(100 * p.true_ohss_rate_high_risk).toFixed(2)}%</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </section>
            )}
          </>
        )}
      </main>
    </div>
  );
}
