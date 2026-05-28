import type { Config } from "tailwindcss";

/**
 * Concordance research palette. Keeps the dashboard dark, with brand red as the
 * primary accent and the pedantic support colors reserved for status/data cues.
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
          950: "#080808",
          900: "#0e1014",
          850: "#12151b",
          800: "#191d25",
          750: "#202632",
          700: "#2b3240",
          600: "#3b4454",
          500: "#596272",
          400: "#858d9a",
          300: "#b7bec8",
          200: "#d9dee6",
          100: "#edf0f5",
          50: "#f8fafc",
        },
        status: {
          ok: "#2E8C43",
          warn: "#F5CD2F",
          fail: "#EF3333",
          run: "#4A6FE0",
          reuse: "#8a68d6",
          idle: "#596272",
        },
        accent: {
          DEFAULT: "#EF3333",
          hot: "#ff6b5f",
          blue: "#4A6FE0",
          green: "#2E8C43",
          yellow: "#F5CD2F",
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
