<template>
  <div class="background-layer" :style="{ '--wallpaper-gradient': theme.activeWallpaper.gradient }">
    <div class="forest-silhouette tree-a"></div>
    <div class="forest-silhouette tree-b"></div>
    <div class="sunshaft shaft-one"></div>
    <div class="sunshaft shaft-two"></div>
    <div class="floating-birds" aria-hidden="true">⌁ ᨒ ︿ ᨒ ⌁</div>
  </div>
</template>

<script setup>
import { onMounted } from 'vue'
import { useThemeStore } from '../stores/theme'

const theme = useThemeStore()

onMounted(() => {
  theme.applyTheme()
})
</script>
