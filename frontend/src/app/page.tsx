"use client";

import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";

type Citation = {
  citation_id: number;
  ticker: string;
  form: string;
  fiscal_year: number;
  item_label: string;
};

type QueryResponse = {
  query: string;
  answer: string;
  confidence: number | null;
  citations: Citation[];
};

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export default function Home() {
  const [query, setQuery] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<QueryResponse | null>(null);

  async function handleSubmit() {
    if (!query.trim() || loading) return;
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const res = await fetch(`${API_URL}/query`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query }),
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
    <div className="flex flex-1 justify-center bg-zinc-50 dark:bg-black">
      <main className="flex w-full max-w-2xl flex-col gap-6 px-6 py-16">
        <div className="space-y-1">
          <h1 className="text-2xl font-semibold tracking-tight">
            Financial RAG
          </h1>
          <p className="text-sm text-muted-foreground">
            Ask a question about AAPL, MSFT, TSLA, GOOGL, or AMZN 10-K/10-Q filings.
          </p>
        </div>

        <div className="flex flex-col gap-3">
          <Textarea
            placeholder="e.g. What was Apple's FY2023 net sales?"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={handleKeyDown}
            rows={3}
            disabled={loading}
          />
          <Button onClick={handleSubmit} disabled={loading || !query.trim()} className="self-end">
            {loading ? "Thinking…" : "Ask"}
          </Button>
        </div>

        {error && (
          <p className="text-sm text-destructive">{error}</p>
        )}

        {loading && (
          <Card>
            <CardContent className="space-y-2">
              <Skeleton className="h-4 w-3/4" />
              <Skeleton className="h-4 w-full" />
              <Skeleton className="h-4 w-1/2" />
            </CardContent>
          </Card>
        )}

        {result && !loading && (
          <Card>
            <CardContent className="space-y-4">
              <p className="leading-relaxed whitespace-pre-wrap">{result.answer}</p>

              {result.confidence !== null && (
                <p className="text-xs text-muted-foreground">
                  Confidence: {result.confidence.toFixed(2)}
                </p>
              )}

              {result.citations.length > 0 && (
                <div className="flex flex-wrap gap-2 border-t pt-3">
                  {result.citations.map((c) => (
                    <Badge key={c.citation_id} variant="secondary">
                      [{c.citation_id}] {c.ticker} {c.form} FY{c.fiscal_year} · {c.item_label}
                    </Badge>
                  ))}
                </div>
              )}
            </CardContent>
          </Card>
        )}
      </main>
    </div>
  );
}
