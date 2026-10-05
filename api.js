async function request(path, options) {
  const response = await fetch(path, options);
  if (!response.ok) {
    let detail = `The server answered ${response.status}.`;
    try {
      const body = await response.json();
      if (typeof body.detail === "string") detail = body.detail;
      else if (Array.isArray(body.detail)) detail = body.detail.map((d) => `${d.loc.slice(-1)[0]}: ${d.msg}`).join("; ");
    } catch (_) {
      /* keep the generic message */
    }
    throw new Error(detail);
  }
  return response.json();
}

const post = (path, body) =>
  request(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });

export const getRecommendation = (patient) => post("/api/recommend", patient);
export const getSimilarPatients = (patient) => post("/api/similar", patient);
export const getPolicies = () => request("/api/policies");
