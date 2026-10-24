/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        obsidian: '#09090B',
        surface: '#18181B',
        'border-sub': '#27272A',
        muted: '#85858F', // ≥4.5:1 on both obsidian and surface
        fg: '#E4E4E7',
        orange: '#F97316',
        success: '#22C55E',
        danger: '#EF4444',
      },
      fontFamily: {
        mono: ['"Geist Mono"', 'ui-monospace', 'SFMono-Regular', 'monospace'],
        sans: ['"Geist"', 'ui-sans-serif', 'system-ui', 'sans-serif'],
      },
      fontSize: {
        '2xs': ['0.6875rem', { lineHeight: '1rem' }],
      },
    },
  },
  plugins: [],
}
