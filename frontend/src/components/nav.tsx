"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { cn } from "@/lib/utils";
import { MessageSquare, BarChart3 } from "lucide-react";

const links = [
  { href: "/", label: "Chat", icon: MessageSquare },
  { href: "/eval", label: "Eval Dashboard", icon: BarChart3 },
];

export function Nav() {
  const pathname = usePathname();

  return (
    <header className="sticky top-0 z-10 border-b-2 border-black bg-background/90 backdrop-blur-sm">
      <div className="mx-auto flex h-16 w-full max-w-4xl items-center justify-between px-6">
        <span className="text-sm font-bold tracking-tight">
          Financial <span className="rounded bg-yellow-300 px-1">RAG</span>
        </span>
        <nav className="flex items-center gap-2">
          {links.map(({ href, label, icon: Icon }) => {
            const active = pathname === href;
            return (
              <Link
                key={href}
                href={href}
                className={cn(
                  "flex items-center gap-1.5 rounded-full border-2 px-3 py-1.5 text-sm font-semibold transition-all",
                  active
                    ? "border-black bg-pink-300 text-black shadow-[2px_2px_0_0_#000]"
                    : "border-transparent text-muted-foreground hover:border-black hover:bg-secondary/60 hover:text-foreground"
                )}
              >
                <Icon className="size-4" />
                {label}
              </Link>
            );
          })}
        </nav>
      </div>
    </header>
  );
}
