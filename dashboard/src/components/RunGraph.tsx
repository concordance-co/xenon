import { useEffect, useMemo, useState } from "react";
import ReactFlow, {
  Background,
  Controls,
  Handle,
  MiniMap,
  Position,
  type Edge,
  type Node,
  type NodeProps,
  MarkerType,
} from "reactflow";
import dagre from "dagre";
import type { DagEdge, DagNode, StepStatus } from "@/types/api";
import { familyAccent, statusBorderClass, statusDotClass } from "@/lib/status";

/**
 * Laid-out DAG for a single run. React Flow + Dagre. Layout is recomputed when
 * nodes/edges change; graph surfaces the resolved-vs-declared edge distinction
 * by dashing resolved-only edges.
 */
export function RunGraph({
  nodes,
  edges,
  selected,
  onSelect,
}: {
  nodes: DagNode[];
  edges: DagEdge[];
  selected: string | null;
  onSelect: (stepName: string) => void;
}) {
  const [hover, setHover] = useState<string | null>(null);
  const { rfNodes, rfEdges } = useMemo(() => layoutDag(nodes, edges), [nodes, edges]);

  // Augment nodes with selection + hover data and attach our custom renderer.
  const neighbors = useMemo(() => {
    const focus = hover ?? selected;
    if (!focus) return null;
    const up = new Set<string>();
    const down = new Set<string>();
    for (const e of edges) {
      if (e.target === focus) up.add(e.source);
      if (e.source === focus) down.add(e.target);
    }
    return { focus, up, down };
  }, [edges, hover, selected]);

  const styledNodes: Node[] = rfNodes.map((n) => ({
    ...n,
    data: {
      ...n.data,
      selected: n.id === selected,
      focus: neighbors && neighbors.focus === n.id,
      upstream: neighbors ? neighbors.up.has(n.id) : false,
      downstream: neighbors ? neighbors.down.has(n.id) : false,
      dimmed:
        neighbors !== null &&
        n.id !== neighbors.focus &&
        !neighbors.up.has(n.id) &&
        !neighbors.down.has(n.id),
    },
  }));

  const styledEdges: Edge[] = rfEdges.map((e) => {
    const isNeighborEdge =
      neighbors !== null &&
      (e.source === neighbors.focus || e.target === neighbors.focus);
    return {
      ...e,
      animated: isNeighborEdge,
      className:
        (e.data?.kind === "declared" ? "" : "react-flow__edge-resolved") +
        (isNeighborEdge ? " react-flow__edge-highlighted" : ""),
      style: {
        ...(e.style ?? {}),
        opacity: neighbors && !isNeighborEdge ? 0.35 : 1,
      },
    };
  });

  return (
    <ReactFlow
      nodes={styledNodes}
      edges={styledEdges}
      nodeTypes={NODE_TYPES}
      onNodeClick={(_, node) => onSelect(node.id)}
      onNodeMouseEnter={(_, node) => setHover(node.id)}
      onNodeMouseLeave={() => setHover(null)}
      fitView
      fitViewOptions={{ padding: 0.25, maxZoom: 1.1 }}
      minZoom={0.35}
      maxZoom={1.5}
      proOptions={{ hideAttribution: true }}
      panOnScroll
      selectionOnDrag
      elementsSelectable
      nodesDraggable={false}
      nodesConnectable={false}
    >
      <Background color="#322e27" gap={24} size={1} />
      <MiniMap
        nodeColor={(n) => miniColor((n.data as GraphNodeData | undefined)?.status)}
        pannable
        zoomable
      />
      <Controls showInteractive={false} />
    </ReactFlow>
  );
}

interface GraphNodeData extends DagNode {
  selected?: boolean;
  focus?: boolean;
  upstream?: boolean;
  downstream?: boolean;
  dimmed?: boolean;
}

const NODE_TYPES = { step: StepNode };

function StepNode(props: NodeProps<GraphNodeData>) {
  const { data } = props;
  const accent = familyAccent(data.family);
  return (
    <div
      className={[
        "group relative min-w-[12.5rem] max-w-[13rem] border bg-ink-850 transition-opacity shadow-[0_1px_0_0_rgba(0,0,0,0.6)]",
        statusBorderClass(data.status as StepStatus),
        data.selected ? "ring-2 ring-accent ring-offset-2 ring-offset-ink-950" : "",
        data.focus && !data.selected ? "ring-1 ring-accent/60" : "",
        data.upstream && !data.focus ? "ring-1 ring-ink-400/50" : "",
        data.downstream && !data.focus ? "ring-1 ring-ink-400/50" : "",
        data.dimmed ? "opacity-35" : "opacity-100",
      ].join(" ")}
    >
      <Handle type="target" position={Position.Top} isConnectable={false} />

      {/* Family accent bar — 3px, spans the full width */}
      <div className={`h-[3px] w-full ${accent.bar}`} />

      {/* Top row: family code chip + index slot. Creates the "model-number" feel. */}
      <div className="flex items-center justify-between px-2 py-1 border-b border-ink-800 bg-ink-900">
        <span className={`text-[0.58rem] font-mono uppercase tracking-[0.18em] ${accent.text}`}>
          {accent.label}
        </span>
        <span className="flex items-center gap-1 text-[0.58rem] font-mono uppercase tracking-wider text-ink-500">
          {data.reused ? (
            <span className="text-status-reuse border border-status-reuse/40 px-1 leading-tight rounded-[1px]">
              RE·USE
            </span>
          ) : null}
          {data.runtime_app_id ? (
            <span className="text-status-run border border-status-run/40 px-1 leading-tight rounded-[1px]">
              MODAL
            </span>
          ) : null}
        </span>
      </div>

      {/* Name + status line */}
      <div className="px-2 py-1.5 flex items-center gap-1.5">
        <span className={`dot ${statusDotClass(data.status)}`} />
        <span className="mono text-xs font-semibold text-ink-50 truncate flex-1">
          {data.step_name}
        </span>
      </div>

      {/* Secondary info block — runner + artifact, with hairline divider */}
      <div className="px-2 pb-1.5 space-y-0.5 text-[0.625rem] font-mono">
        <div className="flex items-center justify-between gap-2 text-ink-400">
          <span className="text-ink-600 tracking-wider">rnr</span>
          <span className="truncate" title={data.runner}>
            {data.runner}
          </span>
        </div>
        {data.artifact_id ? (
          <div className="flex items-center justify-between gap-2 text-ink-500">
            <span className="text-ink-600 tracking-wider">aid</span>
            <span className="truncate" title={data.artifact_id}>
              {data.artifact_id.slice(0, 16)}
            </span>
          </div>
        ) : null}
      </div>

      {/* Corner tick — subtle machine-plate detail */}
      <span className="absolute top-0 right-0 w-2 h-2 border-t border-r border-ink-700 pointer-events-none" />

      <Handle type="source" position={Position.Bottom} isConnectable={false} />
    </div>
  );
}

function miniColor(status: string | undefined): string {
  switch (status) {
    case "completed":
      return "#7fb069";
    case "failed":
      return "#d4675a";
    case "running":
      return "#6ea8c9";
    case "reused":
      return "#a384c4";
    case "pending":
      return "#44403a";
    default:
      return "#615c54";
  }
}

// ---------------------------------------------------------------------------
// Layout

const NODE_W = 208;
const NODE_H = 112;

function layoutDag(nodes: DagNode[], edges: DagEdge[]): { rfNodes: Node[]; rfEdges: Edge[] } {
  const g = new dagre.graphlib.Graph();
  g.setGraph({ rankdir: "TB", nodesep: 32, ranksep: 48, marginx: 16, marginy: 16 });
  g.setDefaultEdgeLabel(() => ({}));

  for (const n of nodes) g.setNode(n.id, { width: NODE_W, height: NODE_H });
  for (const e of edges) g.setEdge(e.source, e.target);

  try {
    dagre.layout(g);
  } catch {
    // fallback: linear rank so we still render something
    nodes.forEach((n, idx) =>
      g.setNode(n.id, { width: NODE_W, height: NODE_H, x: 0, y: idx * (NODE_H + 24) }),
    );
  }

  const rfNodes: Node[] = nodes.map((n) => {
    const pos = g.node(n.id);
    return {
      id: n.id,
      type: "step",
      position: { x: pos?.x ? pos.x - NODE_W / 2 : 0, y: pos?.y ? pos.y - NODE_H / 2 : 0 },
      data: { ...n },
      draggable: false,
      selectable: true,
    } satisfies Node;
  });

  const rfEdges: Edge[] = edges.map((e, idx) => ({
    id: `e_${idx}_${e.source}__${e.target}`,
    source: e.source,
    target: e.target,
    type: "smoothstep",
    data: { kind: e.kind },
    markerEnd: {
      type: MarkerType.ArrowClosed,
      width: 12,
      height: 12,
      color: "#615c54",
    },
  }));

  return { rfNodes, rfEdges };
}

/** Best-effort hook: re-fit when node set changes drastically (e.g. route nav). */
export function useFitOnNodesChange(_nodesLength: number) {
  useEffect(() => {
    // no-op; React Flow's fitView prop handles initial fit.
  }, [_nodesLength]);
}
