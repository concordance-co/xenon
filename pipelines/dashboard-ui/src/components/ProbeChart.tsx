import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ReferenceLine,
} from 'recharts'
import type { ProbeRow } from '../types/api'

interface Props {
  data: ProbeRow[]
}

export function ProbeChart({ data }: Props) {
  // Find layers with high selectivity for reference lines
  const hotLayers = data.filter(r => r.selectivity > 0.05).map(r => r.layer)

  // Detect color scheme
  const isDark = typeof window !== 'undefined' && window.matchMedia('(prefers-color-scheme: dark)').matches

  const gridColor = isDark ? 'rgba(255,255,255,0.06)' : 'rgba(0,0,0,0.06)'
  const textColor = isDark ? '#8b8f9a' : '#5b6170'

  return (
    <div style={{ width: '100%', height: 340 }}>
      <ResponsiveContainer>
        <LineChart data={data} margin={{ top: 8, right: 12, left: 0, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke={gridColor} />

          <XAxis
            dataKey="layer"
            tick={{ fontSize: 11, fill: textColor, fontFamily: "'Fragment Mono', monospace" }}
            label={{ value: 'Layer', position: 'insideBottomRight', offset: -4, fontSize: 11, fill: textColor }}
          />
          <YAxis
            domain={[0, 1]}
            tick={{ fontSize: 11, fill: textColor, fontFamily: "'Fragment Mono', monospace" }}
            tickFormatter={(v: number) => v.toFixed(1)}
            width={36}
          />

          <Tooltip
            contentStyle={{
              background: isDark ? '#1c1d24' : 'white',
              border: `1px solid ${isDark ? 'rgba(255,255,255,0.1)' : 'rgba(0,0,0,0.1)'}`,
              borderRadius: 6,
              fontSize: 12,
              fontFamily: "'Fragment Mono', monospace",
            }}
            // eslint-disable-next-line @typescript-eslint/no-explicit-any
            formatter={((value: any, name: any) => [Number(value).toFixed(4), name]) as any}
            labelFormatter={(layer) => `Layer ${layer}`}
          />

          <Legend
            verticalAlign="top"
            height={28}
            wrapperStyle={{ fontSize: 11, fontFamily: "'Instrument Sans', sans-serif" }}
          />

          {/* Highlight selective layers */}
          {hotLayers.map(layer => (
            <ReferenceLine
              key={layer}
              x={layer}
              stroke={isDark ? 'rgba(45,212,191,0.15)' : 'rgba(15,118,110,0.1)'}
              strokeWidth={20}
            />
          ))}

          <Line
            type="monotone"
            dataKey="baseline_majority"
            stroke={isDark ? '#444' : '#bbb'}
            strokeDasharray="8 4"
            dot={false}
            name="Majority Baseline"
            strokeWidth={1}
          />

          <Line
            type="monotone"
            dataKey="baseline_shuffled"
            stroke={isDark ? '#3a3a3a' : '#ddd'}
            strokeDasharray="3 3"
            dot={false}
            name="Shuffled Control"
            strokeWidth={1}
          />

          <Line
            type="monotone"
            dataKey="balanced_accuracy"
            stroke={isDark ? '#f59e0b' : '#b45309'}
            strokeWidth={1.5}
            dot={{ r: 2, fill: isDark ? '#f59e0b' : '#b45309' }}
            name="Balanced Accuracy"
          />

          <Line
            type="monotone"
            dataKey="accuracy_mean"
            stroke={isDark ? '#2dd4bf' : '#0f766e'}
            strokeWidth={2}
            dot={{ r: 2.5, fill: isDark ? '#2dd4bf' : '#0f766e' }}
            activeDot={{ r: 4 }}
            name="Accuracy"
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}
