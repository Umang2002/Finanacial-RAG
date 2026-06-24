import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Gauge } from "lucide-react";

type RetrievalMetrics = {
  hit_rate: number;
  mrr: number;
  ndcg: number;
  precision_at_k: number;
  recall_at_k: number;
};

type RagasMetrics = {
  faithfulness: number | null;
  answer_relevancy: number | null;
  context_precision: number | null;
  context_recall: number | null;
};

type ExperimentRun = {
  config_name: string;
  num_examples: number;
  generation_provider: string | null;
  generation_model: string | null;
  retrieval: RetrievalMetrics;
  ragas: RagasMetrics;
};

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

function fmt(value: number | null): string {
  return value === null ? "—" : value.toFixed(2);
}

function MetricCard({ label, value }: { label: string; value: number | null }) {
  return (
    <Card>
      <CardContent className="flex flex-col gap-1 p-4">
        <span className="text-xs text-muted-foreground">{label}</span>
        <span className="text-2xl font-semibold tabular-nums">{fmt(value)}</span>
      </CardContent>
    </Card>
  );
}

async function getRuns(): Promise<ExperimentRun[]> {
  try {
    const res = await fetch(`${API_URL}/eval/summary`, { cache: "no-store" });
    if (!res.ok) return [];
    const data = await res.json();
    return data.runs ?? [];
  } catch {
    return [];
  }
}

export default async function EvalDashboard() {
  const runs = await getRuns();
  const latest = runs[0] ?? null;

  return (
    <div className="flex flex-1 justify-center bg-zinc-50 dark:bg-black">
      <main className="flex w-full max-w-4xl flex-col gap-6 px-6 py-12">
        <div className="space-y-1.5">
          <div className="flex items-center gap-2">
            <Gauge className="size-5 text-muted-foreground" />
            <h1 className="text-2xl font-semibold tracking-tight">Evaluation Dashboard</h1>
          </div>
          <p className="text-sm text-muted-foreground">
            RAGAS + retrieval metrics logged by{" "}
            <code className="rounded bg-secondary px-1 py-0.5 text-xs">scripts/run_eval.py</code>{" "}
            against FinanceBench.
          </p>
        </div>

        {runs.length === 0 ? (
          <Card>
            <CardContent className="py-10 text-center text-sm text-muted-foreground">
              No experiment runs logged yet. Run{" "}
              <code className="rounded bg-secondary px-1 py-0.5">python scripts/run_eval.py</code>{" "}
              to populate this dashboard.
            </CardContent>
          </Card>
        ) : (
          <>
            <div className="space-y-2">
              <div className="flex items-center gap-2 text-sm font-medium">
                Latest run
                <Badge variant="secondary">{latest!.config_name}</Badge>
                {latest!.generation_model && (
                  <Badge variant="outline">{latest!.generation_model}</Badge>
                )}
              </div>
              <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
                <MetricCard label="Faithfulness" value={latest!.ragas.faithfulness} />
                <MetricCard label="Answer Relevancy" value={latest!.ragas.answer_relevancy} />
                <MetricCard label="Context Precision" value={latest!.ragas.context_precision} />
                <MetricCard label="Context Recall" value={latest!.ragas.context_recall} />
                <MetricCard label="Hit Rate" value={latest!.retrieval.hit_rate} />
                <MetricCard label="MRR" value={latest!.retrieval.mrr} />
                <MetricCard label="NDCG" value={latest!.retrieval.ndcg} />
                <MetricCard label="Precision@K" value={latest!.retrieval.precision_at_k} />
              </div>
            </div>

            <Card>
              <CardContent className="p-0">
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>Experiment</TableHead>
                      <TableHead>Model</TableHead>
                      <TableHead className="text-right">n</TableHead>
                      <TableHead className="text-right">Faithfulness</TableHead>
                      <TableHead className="text-right">Ans. Relevancy</TableHead>
                      <TableHead className="text-right">Ctx. Precision</TableHead>
                      <TableHead className="text-right">Ctx. Recall</TableHead>
                      <TableHead className="text-right">Hit Rate</TableHead>
                      <TableHead className="text-right">MRR</TableHead>
                      <TableHead className="text-right">NDCG</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {runs.map((run, i) => (
                      <TableRow key={`${run.config_name}-${i}`}>
                        <TableCell className="font-medium">{run.config_name}</TableCell>
                        <TableCell className="text-xs text-muted-foreground">
                          {run.generation_provider ?? "—"}/{run.generation_model ?? "—"}
                        </TableCell>
                        <TableCell className="text-right">{run.num_examples}</TableCell>
                        <TableCell className="text-right">{fmt(run.ragas.faithfulness)}</TableCell>
                        <TableCell className="text-right">
                          {fmt(run.ragas.answer_relevancy)}
                        </TableCell>
                        <TableCell className="text-right">
                          {fmt(run.ragas.context_precision)}
                        </TableCell>
                        <TableCell className="text-right">{fmt(run.ragas.context_recall)}</TableCell>
                        <TableCell className="text-right">{fmt(run.retrieval.hit_rate)}</TableCell>
                        <TableCell className="text-right">{fmt(run.retrieval.mrr)}</TableCell>
                        <TableCell className="text-right">{fmt(run.retrieval.ndcg)}</TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </CardContent>
            </Card>
          </>
        )}
      </main>
    </div>
  );
}
