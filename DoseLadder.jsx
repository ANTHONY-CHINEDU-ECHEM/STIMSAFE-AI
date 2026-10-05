import React from "react";

// The dose ladder: one column per starting dose. Each column shows the expected oocyte
// yield with its 80 percent range against the 8 to 18 target band, and the OHSS risk beneath.
const WIDTH = 720;
const HEIGHT = 300;
const TOP = 24;
const BOTTOM = 210;
const LEFT = 56;
const MAX_OOCYTES = 36;

const yFor = (value) => BOTTOM - (Math.min(value, MAX_OOCYTES) / MAX_OOCYTES) * (BOTTOM - TOP);

export default function DoseLadder({ options, recommendedArm, riskCeiling }) {
  const step = (WIDTH - LEFT - 16) / options.length;
  return (
    <svg className="ladder" viewBox={`0 0 ${WIDTH} ${HEIGHT}`} role="img"
      aria-label="Expected oocyte yield and OHSS risk for each starting dose">
      <rect x={LEFT} y={yFor(18)} width={WIDTH - LEFT - 16} height={yFor(8) - yFor(18)} className="ladder-target" />
      <text x={WIDTH - 20} y={yFor(18) + 14} textAnchor="end" className="ladder-note">target yield, 8 to 18 oocytes</text>
      {[0, 10, 20, 30].map((tick) => (
        <g key={tick}>
          <line x1={LEFT} x2={WIDTH - 16} y1={yFor(tick)} y2={yFor(tick)} className="ladder-grid" />
          <text x={LEFT - 10} y={yFor(tick) + 4} textAnchor="end" className="ladder-axis">{tick}</text>
        </g>
      ))}
      <text x={14} y={(TOP + BOTTOM) / 2} transform={`rotate(-90 14 ${(TOP + BOTTOM) / 2})`} textAnchor="middle" className="ladder-axis">
        oocytes retrieved
      </text>
      {options.map((option, i) => {
        const x = LEFT + step * (i + 0.5);
        const chosen = option.arm === recommendedArm;
        const tone = !option.allowed ? "blocked" : chosen ? "chosen" : "open";
        const risky = option.ohss_risk > riskCeiling;
        return (
          <g key={option.arm} className={`ladder-col ${tone}`}>
            {chosen && <rect x={x - step / 2 + 6} y={TOP - 14} width={step - 12} height={HEIGHT - TOP + 8} rx="3" className="ladder-chosen" />}
            <line x1={x} x2={x} y1={yFor(option.oocytes_high)} y2={yFor(option.oocytes_low)} className="ladder-range" />
            <circle cx={x} cy={yFor(option.expected_oocytes)} r={chosen ? 8 : 6} className="ladder-dot" />
            <text x={x + 14} y={yFor(option.expected_oocytes) + 4} className="ladder-value">{option.expected_oocytes.toFixed(1)}</text>
            <text x={x} y={BOTTOM + 28} textAnchor="middle" className="ladder-dose">{option.dose_iu} IU</text>
            <text x={x} y={BOTTOM + 50} textAnchor="middle" className={`ladder-risk ${risky ? "risky" : ""}`}>
              OHSS {(100 * option.ohss_risk).toFixed(1)}%
            </text>
            <text x={x} y={BOTTOM + 70} textAnchor="middle" className="ladder-status">
              {chosen ? "recommended" : option.allowed ? "" : "not offered"}
            </text>
          </g>
        );
      })}
    </svg>
  );
}
