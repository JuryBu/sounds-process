<template>
  <div class="modal-scrim" @click.self="$emit('close')">
    <section class="settings-modal glass-panel" role="dialog" aria-modal="true" aria-label="设置">
      <aside class="settings-tabs">
        <button class="tab active" type="button">
          <span>◌</span>
          外观
        </button>
        <button class="tab" type="button">
          <span>▧</span>
          模型
        </button>
      </aside>

      <div class="settings-body">
        <button class="close-btn" type="button" aria-label="关闭设置" @click="$emit('close')">×</button>

        <section class="settings-section">
          <h2>主题模式</h2>
          <div class="segmented">
            <button :class="{ active: theme.darkMode }" type="button" @click="theme.setMode(true)">☾ 深色模式</button>
            <button :class="{ active: !theme.darkMode }" type="button" @click="theme.setMode(false)">☼ 浅色模式</button>
          </div>
        </section>

        <section class="settings-section">
          <h2>主题色</h2>
          <div class="theme-swatches">
            <button
              v-for="(item, key) in theme.themes"
              :key="key"
              class="swatch"
              :class="{ active: theme.currentTheme === key }"
              :style="{ '--swatch-color': item.primary }"
              type="button"
              @click="theme.setTheme(key)"
            >
              <span>{{ theme.currentTheme === key ? '✓' : '' }}</span>
              <em>{{ item.label }}</em>
            </button>
          </div>
        </section>

        <section class="settings-section">
          <h2>预览效果</h2>
          <div class="preview-actions">
            <button class="btn-primary" type="button">主按钮</button>
            <button class="btn-ghost" type="button">次按钮</button>
          </div>
        </section>

        <section class="settings-section">
          <h2>背景管理</h2>
          <div class="wallpaper-grid">
            <button
              v-for="item in theme.wallpapers"
              :key="item.id"
              class="wallpaper-tile"
              :class="{ active: theme.wallpaper === item.id }"
              :style="{ background: item.gradient }"
              type="button"
              @click="theme.setWallpaper(item.id)"
            >
              <span>{{ item.label }}</span>
              <i v-if="theme.wallpaper === item.id">✓</i>
            </button>
          </div>
        </section>

        <section class="settings-section sliders">
          <label>
            <span>叠加层透明度 {{ Math.round(theme.overlayOpacity * 100) }}%</span>
            <input type="range" min="0.3" max="0.82" step="0.01" :value="theme.overlayOpacity" @input="theme.setOverlay($event.target.value)" />
          </label>
          <label>
            <span>磨砂模糊 {{ theme.blurAmount }}px</span>
            <input type="range" min="8" max="28" step="1" :value="theme.blurAmount" @input="theme.setBlur($event.target.value)" />
          </label>
        </section>

        <footer class="settings-footer">
          <button class="btn-primary done-btn" type="button" @click="$emit('close')">完成</button>
        </footer>
      </div>
    </section>
  </div>
</template>

<script setup>
import { useThemeStore } from '../stores/theme'

defineEmits(['close'])

const theme = useThemeStore()
</script>
