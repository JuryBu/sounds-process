<template>
  <section class="results-grid" aria-label="Top-5 识别结果">
    <article
      v-for="(bird, index) in displayBirds"
      :key="bird.scientific_name || bird.scientific"
      class="glass-panel result-card"
      :class="{ featured: index === 0 }"
      :style="{ '--delay': `${index * 90}ms` }"
    >
      <div class="bird-portrait" :style="{ background: bird.gradient || gradients[index % gradients.length] }">
        <span>{{ bird.badge }}</span>
      </div>
      <div class="result-copy">
        <span class="rank">Top {{ index + 1 }}</span>
        <h3>{{ bird.common_name || bird.name }}</h3>
        <p>{{ bird.scientific_name || bird.scientific }}</p>
        <div class="confidence-row">
          <span>置信度</span>
          <strong>{{ confidencePercent(bird.confidence) }}%</strong>
        </div>
        <div class="confidence-track">
          <i :style="{ width: `${confidencePercent(bird.confidence)}%` }"></i>
        </div>
      </div>
    </article>
  </section>
</template>

<script setup>
import { computed } from 'vue'

const props = defineProps({
  items: {
    type: Array,
    default: null,
  },
})

const gradients = [
  'radial-gradient(circle at 58% 36%, #ffcf80 0 7%, transparent 8%), linear-gradient(135deg, #111 0%, #2c2d24 45%, #c0732b 100%)',
  'linear-gradient(135deg, #12213d 0%, #497ed3 48%, #e9f4ff 100%)',
  'linear-gradient(135deg, #1b1b1e 0%, #bc542f 48%, #f0c376 100%)',
  'linear-gradient(135deg, #111710 0%, #e8dc70 42%, #4b6d35 100%)',
  'linear-gradient(135deg, #283a32 0%, #d8dfd7 50%, #6a7d54 100%)',
]
const displayBirds = computed(() => {
  if (!props.items?.length) return []
  return props.items.map((item, index) => ({ ...item, badge: ['●', '◆', '▲', '◐', '◇'][index] || '•' }))
})

function confidencePercent(value) {
  const numeric = Number(value || 0)
  return Math.round(numeric <= 1 ? numeric * 100 : numeric)
}
</script>
