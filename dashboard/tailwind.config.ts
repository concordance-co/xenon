import type { Config } from "tailwindcss";

/**
 * Industrial/utilitarian palette. Warm dark neutrals with sharp status accents.
 * Intentionally NOT the default "glassy AI dashboard" look — slate/gray replaced
 * with warm stone, status accents kept saturated.
 */
const config: Config = {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      fontFamily: {
        sans: [
          "'IBM Plex Sans'",
          "ui-sans-serif",
          "system-ui",
          "sans-serif",
        ],
        mono: [
          "'JetBrains Mono'",
          "'IBM Plex Mono'",
          "ui-monospace",
          "SFMono-Regular",
          "Menlo",
          "Consolas",
          "monospace",
        ],
      },
      colors: {
        ink: {
          950: "#0c0b09",
          900: "#15130f",
          850: "#1a1814",
          800: "#221f1a",
          750: "#2a2621",
          700: "#322e27",
          600: "#44403a",
          500: "#615c54",
          400: "#8c867c",
          300: "#b7b1a5",
          200: "#d7d1c4",
          100: "#ebe7db",
          50: "#f6f3ea",
        },
        status: {
          ok: "#7fb069",
          warn: "#e0a458",
          fail: "#d4675a",
          run: "#6ea8c9",
          reuse: "#a384c4",
          idle: "#615c54",
        },
        accent: {
          DEFAULT: "#e0a458",
          hot: "#d4675a",
        },
      },
      boxShadow: {
        inspector: "inset 1px 0 0 rgba(255,255,255,0.04)",
        rail: "inset -1px 0 0 rgba(255,255,255,0.04)",
      },
      fontSize: {
        "2xs": ["0.6875rem", { lineHeight: "1rem" }],
      },
    },
  },
  plugins: [],
};
export default config;
