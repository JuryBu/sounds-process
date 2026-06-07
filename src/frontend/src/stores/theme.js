import { defineStore } from 'pinia'

const themeMap = {
  violet: { label: '紫罗兰', primary: '#a78bfa', glow: 'rgba(167, 139, 250, 0.38)' },
  sky: { label: '天空蓝', primary: '#38bdf8', glow: 'rgba(56, 189, 248, 0.3)' },
  emerald: { label: '翠绿', primary: '#4ade80', glow: 'rgba(74, 222, 128, 0.34)' },
  pink: { label: '樱花粉', primary: '#f472b6', glow: 'rgba(244, 114, 182, 0.32)' },
  orange: { label: '暖橙', primary: '#fb923c', glow: 'rgba(251, 146, 60, 0.34)' },
}

const wallpapers = [
  { id: 'forest-dawn', label: '晨雾森林', gradient: 'radial-gradient(circle at 76% 14%, rgba(255, 222, 155, 0.34), transparent 24%), linear-gradient(135deg, #07110b 0%, #173824 48%, #07100d 100%)' },
  { id: 'moss-river', label: '苔石溪谷', gradient: 'radial-gradient(circle at 24% 22%, rgba(95, 255, 169, 0.22), transparent 27%), linear-gradient(145deg, #06100f 0%, #12382e 42%, #091114 100%)' },
  { id: 'pine-rain', label: '冷杉雨后', gradient: 'radial-gradient(circle at 70% 38%, rgba(120, 196, 255, 0.2), transparent 25%), linear-gradient(150deg, #071018 0%, #123033 52%, #06090a 100%)' },
  { id: 'sun-creek', label: '暖阳溪径', gradient: 'radial-gradient(circle at 62% 18%, rgba(255, 183, 93, 0.38), transparent 23%), linear-gradient(135deg, #0a1309 0%, #24401a 50%, #0c0b06 100%)' },
  { id: 'night-canopy', label: '夜色树冠', gradient: 'radial-gradient(circle at 36% 20%, rgba(108, 255, 184, 0.18), transparent 21%), linear-gradient(150deg, #030807 0%, #0b1f1a 46%, #020404 100%)' },
]

export const useThemeStore = defineStore('theme', {
  state: () => ({
    currentTheme: localStorage.getItem('birdvoice-theme') || 'emerald',
    darkMode: localStorage.getItem('birdvoice-mode') !== 'light',
    wallpaper: localStorage.getItem('birdvoice-wallpaper') || 'forest-dawn',
    overlayOpacity: Number(localStorage.getItem('birdvoice-overlay') || 0.62),
    blurAmount: Number(localStorage.getItem('birdvoice-blur') || 18),
    themes: themeMap,
    wallpapers,
  }),
  getters: {
    activeTheme: (state) => state.themes[state.currentTheme],
    activeWallpaper: (state) => state.wallpapers.find((item) => item.id === state.wallpaper) || state.wallpapers[0],
  },
  actions: {
    applyTheme() {
      const theme = this.activeTheme
      document.documentElement.style.setProperty('--primary', theme.primary)
      document.documentElement.style.setProperty('--primary-glow', theme.glow)
      document.documentElement.style.setProperty('--overlay-opacity', this.overlayOpacity)
      document.documentElement.style.setProperty('--scene-blur', `${this.blurAmount}px`)
      document.documentElement.dataset.mode = this.darkMode ? 'dark' : 'light'
    },
    setTheme(name) {
      this.currentTheme = name
      localStorage.setItem('birdvoice-theme', name)
      this.applyTheme()
    },
    setMode(isDark) {
      this.darkMode = isDark
      localStorage.setItem('birdvoice-mode', isDark ? 'dark' : 'light')
      this.applyTheme()
    },
    setWallpaper(id) {
      this.wallpaper = id
      localStorage.setItem('birdvoice-wallpaper', id)
    },
    setOverlay(value) {
      this.overlayOpacity = Number(value)
      localStorage.setItem('birdvoice-overlay', this.overlayOpacity)
      this.applyTheme()
    },
    setBlur(value) {
      this.blurAmount = Number(value)
      localStorage.setItem('birdvoice-blur', this.blurAmount)
      this.applyTheme()
    },
  },
})
