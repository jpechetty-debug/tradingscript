import React from 'react';
import { probabilityEvidence } from '../evidence';

export default function ProbabilityEvidence({ item }) {
  const evidence = probabilityEvidence(item);
  return (
    <div className={`probability-evidence ${evidence.tone}`}>
      <strong className="mono">{evidence.text}</strong>
      <span>{evidence.label}</span>
    </div>
  );
}

export function StrategyEvidence({ item }) {
  return <span className="strategy-evidence">Strategy: {item.strategy_id || 'Not reported'}
    {item.strategy_version && ` · ${item.strategy_version}`}</span>;
}
