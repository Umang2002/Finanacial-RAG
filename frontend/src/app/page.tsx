"use client";

import { useEffect, useState } from "react";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { Switch } from "@/components/ui/switch";
import { Separator } from "@/components/ui/separator";
import {
  Tabs,
  TabsContent,
  TabsList,
  TabsTrigger,
} from "@/components/ui/tabs";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Info, Loader2, SearchCheck, Sparkles } from "lucide-react";

type Citation = {
  citation_id: number;
  ticker: string;
  form: string;
  fiscal_year: number;
  item_label: string;
};

type DebugHit = {
  score: number;
  ticker: string;
  form: string;
  fiscal_year: number;
  item_label: string;
  text_preview: string;
};

type DebugStages = {
  dense: DebugHit[];
  sparse: DebugHit[];
  hybrid: DebugHit[];
  reranked: DebugHit[];
};

type QueryResponse = {
  query: string;
  answer: string;
  confidence: number | null;
  citations: Citation[];
  debug: DebugStages | null;
};

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

const STICKER_VARIANTS = ["amber", "violet", "emerald", "rose"] as const;

// LEARN: the API is one synchronous request (see CLAUDE.md Phase 8 notes) —
// it doesn't stream real per-stage progress, so this just cycles through the
// pipeline's known steps on a timer to keep the wait from feeling frozen.
const LOADING_STAGES = [
  "Classifying intent…",
  "Expanding the query (HyDE / multi-query)…",
  "Searching dense + sparse indexes…",
  "Reranking candidates…",
  "Generating a cited answer…",
];

const STAGES: { key: keyof DebugStages; label: string }[] = [
  { key: "dense", label: "Dense" },
  { key: "sparse", label: "Sparse" },
  { key: "hybrid", label: "Hybrid (RRF)" },
  { key: "reranked", label: "Reranked" },
];

function DebugStageTable({ hits }: { hits: DebugHit[] }) {
  if (hits.length === 0) {
    return <p className="py-6 text-center text-sm text-muted-foreground">No hits.</p>;
  }
  return (
    <Table>
      <TableHeader>
        <TableRow>
          <TableHead className="w-20">Score</TableHead>
          <TableHead className="w-36">Filing</TableHead>
          <TableHead className="w-28">Section</TableHead>
          <TableHead>Preview</TableHead>
        </TableRow>
      </TableHeader>
      <TableBody>
        {hits.map((hit, i) => (
          <TableRow key={i}>
            <TableCell className="font-mono text-xs">{hit.score.toFixed(4)}</TableCell>
            <TableCell className="text-xs">
              {hit.ticker} {hit.form} FY{hit.fiscal_year}
            </TableCell>
            <TableCell className="text-xs text-muted-foreground">{hit.item_label}</TableCell>
            <TableCell className="text-xs text-muted-foreground whitespace-normal">
              {hit.text_preview}…
            </TableCell>
          </TableRow>
        ))}
      </TableBody>
    </Table>
  );
}

export default function Home() {
  const [query, setQuery] = useState("");
  const [debugEnabled, setDebugEnabled] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<QueryResponse | null>(null);
  const [stageIndex, setStageIndex] = useState(0);
  const [elapsed, setElapsed] = useState(0);

  useEffect(() => {
    if (!loading) return;
    setStageIndex(0);
    setElapsed(0);
    const stageTimer = setInterval(
      () => setStageIndex((i) => Math.min(i + 1, LOADING_STAGES.length - 1)),
      4000
    );
    const clock = setInterval(() => setElapsed((s) => s + 1), 1000);
    return () => {
      clearInterval(stageTimer);
      clearInterval(clock);
    };
  }, [loading]);

  async function handleSubmit() {
    if (!query.trim() || loading) return;
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const res = await fetch(`${API_URL}/query`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query, debug: debugEnabled }),
      });
      if (!res.ok) throw new Error(`Request failed (${res.status})`);
      setResult(await res.json());
    } catch (err) {
      setError(err instanceof Error ? err.message : "Something went wrong");
    } finally {
      setLoading(false);
    }
  }

  function handleKeyDown(e: React.KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  }

  return (
    <div className="flex flex-1 justify-center">
      <main className="flex w-full max-w-2xl flex-col gap-6 px-6 py-12">
        <div className="space-y-1.5">
          <div className="flex items-center gap-2">
            <Sparkles className="size-5" />
            <h1 className="text-2xl font-extrabold tracking-tight">Ask the filings</h1>
          </div>
          <p className="text-sm text-muted-foreground">
            Hybrid search RAG over AAPL, MSFT, TSLA, GOOGL, and AMZN 10-K / 10-Q filings.
          </p>
        </div>

        <Card className="bg-secondary/40">
          <CardContent className="flex gap-2.5 text-sm text-muted-foreground">
            <Info className="size-4 shrink-0 translate-y-0.5" />
            <p>
              Every answer is grounded in real SEC filings (2021–2024): the query is expanded
              (HyDE + multi-query), matched with dense + BM25 search, reranked with a
              cross-encoder, then answered with inline{" "}
              <span className="font-semibold text-foreground">[citations]</span> back to the
              exact filing section. Flip on <span className="font-semibold">Retrieval debug</span>{" "}
              below to see what each stage actually retrieved.
            </p>
          </CardContent>
        </Card>

        <div className="flex flex-col gap-3">
          <Textarea
            placeholder="e.g. What was Apple's FY2023 net sales?"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={handleKeyDown}
            rows={3}
            disabled={loading}
          />
          <div className="flex items-center justify-between">
            <label className="flex items-center gap-2 text-sm font-medium text-muted-foreground">
              <Switch
                checked={debugEnabled}
                onCheckedChange={setDebugEnabled}
                disabled={loading}
              />
              <SearchCheck className="size-4" />
              Retrieval debug
            </label>
            <Button onClick={handleSubmit} disabled={loading || !query.trim()}>
              {loading && <Loader2 className="size-4 animate-spin" />}
              {loading ? "Thinking…" : "Ask"}
            </Button>
          </div>
        </div>

        {error && (
          <p className="rounded-xl border-2 border-black bg-red-200 px-3 py-2 text-sm font-medium text-black shadow-[3px_3px_0_0_#000]">
            {error}
          </p>
        )}

        {loading && (
          <Card>
            <CardContent className="space-y-3">
              <div className="flex items-center gap-2 text-sm font-semibold">
                <Loader2 className="size-4 animate-spin" />
                {LOADING_STAGES[stageIndex]}
                <span className="ml-auto font-mono text-xs text-muted-foreground">
                  {elapsed}s
                </span>
              </div>
              <Skeleton className="h-4 w-3/4" />
              <Skeleton className="h-4 w-full" />
              <Skeleton className="h-4 w-1/2" />
              <p className="text-xs text-muted-foreground">
                Embedding + reranking run locally on this machine — first queries and
                queries with debug on can take well over a minute.
              </p>
            </CardContent>
          </Card>
        )}

        {result && !loading && (
          <div className="flex flex-col gap-4">
            <div className="relative">
              {result.confidence !== null && (
                <Badge
                  variant="rose"
                  className="absolute -top-3 -right-3 rotate-6 shadow-[2px_2px_0_0_#000]"
                >
                  CONF {result.confidence.toFixed(2)}
                </Badge>
              )}
              <Card>
                <div className="flex items-center gap-1.5 border-b-2 border-black px-4 py-2.5">
                  <span className="size-2.5 rounded-full border border-black bg-red-400" />
                  <span className="size-2.5 rounded-full border border-black bg-yellow-300" />
                  <span className="size-2.5 rounded-full border border-black bg-emerald-400" />
                  <span className="ml-2 text-xs font-medium text-muted-foreground">answer</span>
                </div>
                <CardContent className="space-y-4">
                  <p className="leading-relaxed whitespace-pre-wrap">{result.answer}</p>

                  {result.citations.length > 0 && (
                    <>
                      <Separator className="border-black/20" />
                      <div className="flex flex-wrap gap-2">
                        {result.citations.map((c, i) => (
                          <Badge
                            key={c.citation_id}
                            variant={STICKER_VARIANTS[i % STICKER_VARIANTS.length]}
                            className={i % 2 === 0 ? "-rotate-1" : "rotate-1"}
                          >
                            [{c.citation_id}] {c.ticker} {c.form} FY{c.fiscal_year} ·{" "}
                            {c.item_label}
                          </Badge>
                        ))}
                      </div>
                    </>
                  )}
                </CardContent>
              </Card>
            </div>

            {result.debug && (
              <Card>
                <CardContent className="space-y-3">
                  <div className="flex items-center gap-2 text-sm font-medium">
                    <SearchCheck className="size-4 text-muted-foreground" />
                    Retrieval debug
                  </div>
                  <Tabs defaultValue="dense">
                    <TabsList>
                      {STAGES.map((s) => (
                        <TabsTrigger key={s.key} value={s.key}>
                          {s.label}
                        </TabsTrigger>
                      ))}
                    </TabsList>
                    {STAGES.map((s) => (
                      <TabsContent key={s.key} value={s.key}>
                        <DebugStageTable hits={result.debug![s.key]} />
                      </TabsContent>
                    ))}
                  </Tabs>
                </CardContent>
              </Card>
            )}
          </div>
        )}
      </main>
    </div>
  );
}
