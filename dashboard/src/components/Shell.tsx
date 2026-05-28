import { Link, NavLink } from "react-router-dom";
import type { ReactNode } from "react";

/**
 * Thin app shell: single top bar + main content. The left rail was removed
 * because it had exactly one item; navigation happens through the top-bar
 * brand link and in-page back buttons.
 */
export function Shell({ children }: { children: ReactNode }) {
  return (
    <div className="h-screen grid grid-rows-[2.25rem_1fr] bg-ink-950 text-ink-100">
      <TopBar />
      <main className="overflow-hidden relative flex flex-col min-h-0">
        {children}
      </main>
    </div>
  );
}

function TopBar() {
  return (
    <header className="relative flex items-center border-b border-ink-800 bg-ink-900/95">
      <Link
        to="/projects"
        className="flex items-center gap-2 h-full px-3 hover:bg-ink-850 transition-colors group border-r border-ink-800"
      >
        <span className="flex items-center gap-[2px]">
          <span className="w-[3px] h-4 bg-accent" />
          <span className="w-[3px] h-2.5 bg-accent/60" />
          <span className="w-[3px] h-3 bg-accent/30" />
        </span>
        <span className="mono font-semibold tracking-[0.15em] text-[0.7rem] uppercase text-ink-50 group-hover:text-accent">
          pipelines_v2
        </span>
        <span className="text-[0.58rem] uppercase tracking-[0.22em] font-mono text-ink-600 group-hover:text-ink-400">
          research
        </span>
      </Link>
      <nav className="flex h-full items-center border-r border-ink-800">
        <TopNavLink to="/projects">projects</TopNavLink>
        <TopNavLink to="/runs">runs</TopNavLink>
      </nav>
      <div className="absolute bottom-0 right-0 w-16 h-[2px] bg-gradient-to-r from-transparent via-accent/40 to-accent" />
    </header>
  );
}

function TopNavLink({ to, children }: { to: string; children: ReactNode }) {
  return (
    <NavLink
      to={to}
      className={({ isActive }) =>
        [
          "flex h-full items-center border-r border-ink-800 px-3 text-[0.65rem] font-mono uppercase tracking-[0.16em] transition-colors",
          isActive ? "bg-ink-850 text-accent" : "text-ink-500 hover:bg-ink-850 hover:text-ink-100",
        ].join(" ")
      }
    >
      {children}
    </NavLink>
  );
}
