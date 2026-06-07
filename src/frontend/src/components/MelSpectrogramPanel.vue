<template>
  <section class="glass-panel audio-panel mel-panel">
    <div class="panel-heading">
      <div>
        <span class="panel-kicker">Mel Spectrogram</span>
        <h2>梅尔频谱图</h2>
      </div>
      <button v-if="imageSrc" class="expand-btn" type="button" aria-label="放大频谱">↗</button>
    </div>

    <div v-if="imageSrc" class="spectrogram-wrap">
      <div class="freq-axis">
        <span v-for="freq in freqs" :key="freq">{{ freq }}</span>
      </div>
      <img class="mel-image" :src="imageSrc" alt="后端返回的梅尔频谱图" />
    </div>
    <div v-else class="empty-state">
      <span>识别完成后显示梅尔频谱图</span>
    </div>
  </section>
</template>

<script setup>
import { computed } from 'vue'

const props = defineProps({
  mel: {
    type: Object,
    default: null,
  },
})

const freqs = ['8 kHz', '4 kHz', '2 kHz', '1 kHz', '512 Hz', '256 Hz', '128 Hz', '64 Hz']

const imageSrc = computed(() => {
  if (!props.mel?.image_base64) return ''
  return `data:${props.mel.mime || 'image/svg+xml'};base64,${props.mel.image_base64}`
})
</script>
