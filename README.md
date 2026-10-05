# StimSafe AI

**Safety constrained recommendation of ovarian stimulation protocols using contextual bandits**

<p align="center">
  <img width="1380" height="1200" alt="dashboard" src="https://github.com/user-attachments/assets/7d07422b-1f2b-48e6-8cbe-68a69c504629" />

</p>

## Project brief

Every IVF cycle begins with a prescription that has no perfect answer: how much gonadotropin to give. Too little and the ovaries respond poorly, few eggs are collected and the cycle may be cancelled after weeks of injections. Too much and the patient is exposed to ovarian hyperstimulation syndrome (OHSS), a complication that can mean hospital admission and, rarely, a threat to life. The right starting dose depends on ovarian reserve, age, body mass and previous response, and it varies several fold between patients. In most clinics it is still chosen from a mixture of age bands, a glance at AMH and the habits of the prescribing doctor.

What makes this a hard machine learning problem, and not an ordinary prediction task, is that the records only ever show one side of the story. Each historical cycle reveals what happened on the dose that was given and nothing about what would have happened on any other dose. Worse, the doses were not assigned at random: doctors gave low doses to women they expected to respond strongly, so a naive model learns that low doses "cause" high yields. A system that learns from these logs has to reason about decisions and counterfactuals, has to estimate how good a new prescribing policy would be before anyone is treated with it, and has to do all of that without ever trading a patient's safety for a better average.

StimSafe AI is a full stack decision support product built on that framing. It treats protocol selection as a contextual bandit over twelve actions (four starting doses combined with three protocol and trigger regimens), learns outcome models with pharmacological monotonicity constraints, evaluates candidate policies offline with inverse propensity and doubly robust estimators, and wraps the learned policy in an explicit rule based OHSS filter that the model cannot override. A FastAPI backend serves recommendations, per dose predictions and the outcomes of similar past patients, and a React dashboard presents them to the doctor on one screen. The repository also contains a multi output classifier that imitates successful historical cycles, online LinUCB and epsilon greedy learners, tests, containers and continuous integration.

Stimulation records with outcomes are confidential, so the system is trained on 60,000 simulated cycles. The simulator uses a sigmoid Emax dose response model with patient specific capacity and sensitivity, an OHSS risk model driven by yield, and a noisy clinician prescribing heuristic whose choice probabilities are logged. Because the simulator knows each patient's true response to every dose, every policy below is scored against ground truth as well as by the estimators a real clinic would have to rely on. All figures describe simulated patients and none is a clinical claim.

## What the system does

<table>
  <tr><th align="left">Capability</th><th align="left">How it is delivered</th></tr>
  <tr><td>Protocol recommendation</td><td>Greedy policy over a learned reward model for 12 arms: starting dose of 100, 150, 225 or 300 IU with one of three protocol and trigger regimens</td></tr>
  <tr><td>Per dose forecast</td><td>Expected oocyte yield with an 80 percent range and predicted OHSS risk for every arm, constrained to be monotone in dose</td></tr>
  <tr><td>Safety filter</td><td>Configurable clinical thresholds flag high risk patients, cap the dose and restrict the regimen; a predicted risk ceiling applies to everyone</td></tr>
  <tr><td>Multi output classifier</td><td>Three linked classifiers (dose category, protocol, trigger) trained on cycles with a good yield and no OHSS</td></tr>
  <tr><td>Off policy evaluation</td><td>Direct method, inverse propensity scoring, self normalised IPS and doubly robust estimates with bootstrap intervals</td></tr>
  <tr><td>Online learning</td><td>LinUCB with Sherman Morrison updates, with and without the safety filter, against epsilon greedy</td></tr>
  <tr><td>Dashboard</td><td>React front end with a dose ladder chart, safety reasoning, similar patient outcomes and a policy comparison table</td></tr>
</table>

## Key findings

### The constrained policy closes half of the gap between usual practice and a perfect prescriber

On 17,988 held out cycles the logged clinician heuristic earns an expected clinical utility of 0.448 per cycle. A prescriber who knew every patient's true dose response curve would earn 0.637. StimSafe reaches 0.542, recovering 50 percent of that gap, while a single fixed dose (0.382) and an age band rule (0.413) both do worse than the clinicians they might replace.

<table>
  <tr><th align="left">Policy</th><th>Utility per cycle</th><th>Yield in target range</th><th>Poor response</th><th>OHSS rate</th><th>OHSS rate, high risk patients</th></tr>
  <tr><td>Fixed 150 IU for everyone</td><td align="center">0.382</td><td align="center">38.9%</td><td align="center">19.8%</td><td align="center">4.79%</td><td align="center">9.98%</td></tr>
  <tr><td>Age band rule</td><td align="center">0.413</td><td align="center">43.3%</td><td align="center">14.5%</td><td align="center">5.46%</td><td align="center">11.08%</td></tr>
  <tr><td>Clinician heuristic (historical log)</td><td align="center">0.448</td><td align="center">43.0%</td><td align="center">15.7%</td><td align="center">3.38%</td><td align="center">5.17%</td></tr>
  <tr><td>Multi output classifier</td><td align="center">0.484</td><td align="center">44.2%</td><td align="center">15.1%</td><td align="center">2.12%</td><td align="center">2.29%</td></tr>
  <tr><td>Bandit policy, unconstrained</td><td align="center">0.548</td><td align="center">48.5%</td><td align="center">12.1%</td><td align="center">0.71%</td><td align="center">1.38%</td></tr>
  <tr><td><b>StimSafe (bandit with safety filter)</b></td><td align="center"><b>0.542</b></td><td align="center"><b>47.7%</b></td><td align="center"><b>13.0%</b></td><td align="center"><b>0.62%</b></td><td align="center"><b>1.15%</b></td></tr>
  <tr><td>Oracle (true best arm)</td><td align="center">0.637</td><td align="center">54.6%</td><td align="center">8.0%</td><td align="center">0.29%</td><td align="center">0.61%</td></tr>
</table>

<p align="center"><img width="1376" height="688" alt="policy_value" src="https://github.com/user-attachments/assets/fb296045-ca4b-44b8-bcba-f03457427952" /></p>

Translated into a clinic treating 1,000 cycles a year, moving from the historical heuristic to StimSafe corresponds to roughly 28 fewer cases of OHSS (34 down to 6), 47 more cycles with a yield in the 8 to 18 oocyte target range and 27 fewer poor responses. The improvement does not come from prescribing less: the mean recommended dose is 208 IU against 189 IU historically. It comes from prescribing differently, with more drug for low reserve patients and a protective trigger for strong responders.

<p align="center"><img width="2160" height="640" alt="safety_efficacy" src="https://github.com/user-attachments/assets/d2518c7f-7409-41b7-97d8-4a3f80c0e335" /></p>

### Simple rules are less safe than the clinicians they are meant to standardise

It is tempting to replace judgement with a fixed protocol. In this cohort, giving everyone 150 IU nearly doubles the OHSS rate among high risk patients (9.98 percent against 5.17 percent) because it overdoses strong responders, and it also raises poor responses because it underdoses women with low reserve. The age band rule is worse still on safety. Age is a weak proxy for ovarian response; AMH and antral follicle count carry the signal. Any standardisation effort that ignores reserve markers moves risk in the wrong direction.

### The safety filter costs almost nothing and removes the model's worst decisions

The explicit filter flags 38.8 percent of patients as high risk and changes the model's preferred action for 10.3 percent of the cohort. Imposing it lowers utility from 0.548 to 0.542, a cost of about 1 percent, and lowers OHSS among high risk patients from 1.38 percent to 1.15 percent. That is the argument for keeping hard constraints outside the model: the learned policy already behaves well on average, the rules guarantee how it behaves in the tail, and the price of that guarantee is negligible. When every permitted option still exceeds the predicted risk ceiling the system says so and marks the patient for individual review instead of presenting the least bad option as a confident answer.

### Imitating successful cycles helps, but inherits clinician habits

The multi output classifier learns from the 17,866 training cycles that ended with a good yield and no OHSS. It beats the historical heuristic (0.484 against 0.448) and cuts OHSS by more than a third, which makes it a reasonable first step for a clinic without propensity data. Its ceiling is low, though. Its dose agrees with the logged clinician dose in 66 percent of test patients and with the truly optimal dose in only 35 percent. Filtering on success removes the worst decisions; it cannot discover a better dose that clinicians rarely tried. The bandit formulation, which models the outcome of every action, is what closes the larger part of the gap.

### Off policy evaluation ranks realistic policies correctly, with honest limits

A real clinic cannot compute the "true value" column. It has to estimate the value of a new policy from logs. The doubly robust estimator does this well for every deployable policy: each true value falls inside its bootstrap interval.

<table>
  <tr><th align="left">Policy</th><th>True value</th><th>Direct method</th><th>Self normalised IPS</th><th>Doubly robust (95% interval)</th></tr>
  <tr><td>Fixed 150 IU for everyone</td><td align="center">0.382</td><td align="center">0.405</td><td align="center">0.369</td><td align="center">0.380 (0.353 to 0.408)</td></tr>
  <tr><td>Age band rule</td><td align="center">0.413</td><td align="center">0.421</td><td align="center">0.402</td><td align="center">0.404 (0.378 to 0.434)</td></tr>
  <tr><td>Multi output classifier</td><td align="center">0.484</td><td align="center">0.474</td><td align="center">0.492</td><td align="center">0.492 (0.482 to 0.503)</td></tr>
  <tr><td>Bandit policy, unconstrained</td><td align="center">0.548</td><td align="center">0.516</td><td align="center">0.564</td><td align="center">0.579 (0.539 to 0.625)</td></tr>
  <tr><td>StimSafe</td><td align="center">0.542</td><td align="center">0.514</td><td align="center">0.557</td><td align="center">0.569 (0.534 to 0.614)</td></tr>
  <tr><td>Oracle</td><td align="center">0.637</td><td align="center">0.491</td><td align="center">0.535</td><td align="center">0.536 (0.487 to 0.590)</td></tr>
</table>

<p align="center"><img width="992" height="736" alt="ope_accuracy" src="https://github.com/user-attachments/assets/39761dc6-f0ac-412b-ba36-0f4763ef6689" /></p>

Two cautions belong beside that result. The intervals for the bandit policies are four times wider than for the classifier, because the bandit departs further from historical practice: its effective sample size is about 209 cycles out of 17,988. And all three estimators badly underestimate the oracle, whose choices depend on information the reward model does not have. Off policy evaluation is trustworthy for policies near the support of the logs and degrades as a policy becomes more ambitious. The practical consequence is a staged rollout: validate offline, deploy alongside clinicians, then widen.

### The outcome models learn the dose response curve, not the prescribing bias

Monotonic constraints force predicted yield and predicted OHSS risk to rise with dose, which prevents the model from absorbing the confounded pattern in which low doses appear to produce high yields. On logged cycles the yield model has a mean absolute error of 4.5 oocytes against 6.1 for a constant prediction, its 80 percent range covers 80.1 percent of outcomes, and the OHSS model reaches a ROC AUC of 0.85. Scored against the simulator's counterfactual truth across all twelve arms, the mean error in expected yield is 2.6 oocytes.

<p align="center"><img width="2000" height="1024" alt="dose_response" src="https://github.com/user-attachments/assets/7e81c3e2-ea66-4e5d-b51f-15b93095440f" /></p>

### An online bandit halves the regret of fixed practice, and learning safely is cheap

Played forward through the simulator one patient at a time from a cold start, LinUCB accumulates 46 percent less regret than the clinician heuristic over 17,988 cycles (1,813 against 3,373 utility units) and 15 percent less than epsilon greedy. Restricting LinUCB to arms that pass the safety filter raises its regret by under 5 percent. Exploration can therefore be confined to the safe set from the first patient without giving up most of the benefit.

<p align="center"><img width="1184" height="672" alt="online_regret" src="https://github.com/user-attachments/assets/aac97044-668a-48b4-a540-a5309e5e4897" /></p>

### Where the recommendation differs from what was prescribed

<p align="center"><img width="896" height="736" alt="dose_shift" src="https://github.com/user-attachments/assets/7f72ffdb-a9dd-4dde-9758-0bf120196bd2" /></p>

The policy also has a strong view on regimen: 96.7 percent of its recommendations use an antagonist protocol with a GnRH agonist trigger. That reflects the reward function, which penalises OHSS heavily and assigns no cost to the freeze all strategy that an agonist trigger usually implies. A clinic that values fresh transfer would add that cost in `configs/config.yaml` and the policy would shift. The finding is a reminder that a recommender optimises exactly what it is told to value.

## Architecture

```
historical log (context, action, propensity, outcome)
        |
        +==> OutcomeModels           reward, oocyte yield (mean and 80% range), OHSS risk; monotone in dose
        +==> ProtocolClassifier      dose, protocol and trigger learned from successful cycles
        |
        v
candidate policies ==> off policy evaluation (DM, IPS, SNIPS, DR) ==> ground truth check in the simulator
        |
        v
safety filter (clinical thresholds, dose cap, regimen restriction, predicted risk ceiling)
        |
        v
ProtocolRecommender ==> FastAPI  /api/recommend  /api/similar  /api/policies  /api/health
                              |
                              v
                        React dashboard (dose ladder, safety reasoning, similar patients, policy table)
```

## Repository layout

```
stimsafe_ai
    configs/config.yaml               safety thresholds, reward weights, bandit settings
    artifacts/stimsafe_bundle.joblib  trained models, similarity index and policy table
    docs/images                       figures and dashboard screenshot
    frontend                          React and Vite dashboard
        src/App.jsx                   patient form, recommendation, tables
        src/DoseLadder.jsx            SVG dose ladder chart
        src/api.js                    API client with readable error messages
    reports/metrics.json              full evaluation output
    src/stimsafe
        data/schema.py                action space and patient fields
        data/simulate.py              dose response simulator and logging policy
        models/outcome.py             outcome models and the multi output classifier
        policy/safety.py              rule based OHSS filter
        policy/bandit.py              off policy estimators, LinUCB, online simulation
        evaluation/pipeline.py        training, policy evaluation, bundle export
        evaluation/report.py          figure generation
        inference/recommender.py      serving facade
        api/main.py                   FastAPI backend, also serves the built dashboard
    tests                             15 tests covering simulator, safety, estimators, policies and API
    Dockerfile, compose.yaml          two stage image: Node build, Python runtime
```

## Getting started

Python 3.10 or later and Node 20 or later are required.

```
make install
make all
make test
```

`make all` simulates the cohort, trains every model, evaluates every policy and rebuilds the figures in about a minute. A trained bundle is included, so the dashboard also works straight after installation.

Build the dashboard and start the server, then open `http://localhost:8000`:

```
make frontend
make api
```

For front end development with hot reload, run `make api` in one terminal and `make dev` in another, then open `http://localhost:5173`. To run everything in one container:

```
docker compose up
```

### Calling the API

```python
import httpx

patient = {"age": 31, "amh_ng_ml": 4.2, "afc": 19, "fsh_iu_l": 5.6, "bmi": 22.5, "pcos": 0, "first_cycle": 1}
result = httpx.post("http://localhost:8000/api/recommend", json=patient).json()

print(result["recommendation"]["label"])
print(result["safety"]["reasons"])
for option in result["options"]:
    print(option["label"], option["expected_oocytes"], option["ohss_risk"], option["allowed"])
```

## Design decisions

* **Constraints outside the model.** The safety filter is ordinary, readable code driven by a configuration file. It runs after the model and can only remove options.
* **Monotonic outcome models.** Gradient boosting with a monotone constraint on dose encodes the one thing pharmacology guarantees and blocks the most damaging form of confounding.
* **Logged propensities.** The simulator records the probability of each historical choice, as a well instrumented prescribing system should, which is what makes inverse propensity and doubly robust estimation possible.
* **Ground truth kept apart.** Latent patient parameters are written to a separate file that no model reads. They are used only to score policies.
* **Similar patients as context, not evidence.** The dashboard labels historical outcomes as confounded by indication, so the table supports the prediction without being mistaken for a trial.

## Limitations and responsible use

* All data is simulated. The dose response model is a simplification and real patients vary in ways it does not capture.
* The clinical thresholds in the configuration file are illustrative defaults and must be reviewed against local guidelines.
* The utility function is one reasonable encoding of clinical priorities. Different weights give different policies, as the regimen finding shows.
* Off policy estimates are reliable only for policies close to historical practice; overlap should be checked before trusting them.
* StimSafe AI is a research and portfolio prototype. It is not a medical device and must not be used to prescribe.

## Licence

Released under the MIT licence.
