/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        obsidian:      '#09090B',
        surface:       '#18181B',
        'border-sub':  '#27272A',
        muted:         '#71717A',
        fg:            '#E4E4E7',
        orange:        '#F97316',
        'orange-dim':  '#7C3412',
        success:       '#22C55E',
        danger:        '#EF4444',
        info:          '#22D3EE',
        violet:        '#D946EF',
        warn:          '#EAB308',
      },
      fontFamily: {
        mono: ['"Geist Mono"', 'ui-monospace', 'SFMono-Regular', 'monospace'],
        sans: ['"Geist"',      'ui-sans-serif', 'system-ui', 'sans-serif'],
      },
      fontSize: {
        '2xs': ['0.65rem', { lineHeight: '1rem' }],
      },
      animation: {
        'ping-slow': 'ping 1.8s cubic-bezier(0,0,0.2,1) infinite',
      },
    },
  },
  plugins: [],
}
