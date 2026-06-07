<template>
  <section class="history-panel content-grid">
    <section class="upload-dock glass-panel">
      <div class="upload-copy">
        <span>历史记录</span>
        <h2>最近 30 次识别会保存在浏览器 localStorage</h2>
        <p>记录包含文件名、模型、Top-1 结果和置信度，刷新页面后仍可追溯演示过程。</p>
      </div>
      <button class="btn-ghost" type="button" @click="$emit('clear')">清空历史</button>
    </section>

    <section class="glass-panel history-list">
      <article v-for="item in items" :key="item.id" class="history-row">
        <div>
          <span>{{ item.time }}</span>
          <h3>{{ item.filename }}</h3>
          <p>{{ item.model }} · {{ item.winner }}</p>
        </div>
        <strong>{{ confidencePercent(item.confidence) }}%</strong>
      </article>
      <p v-if="!items.length" class="empty-history">还没有识别历史，上传或录音后这里会自动出现记录。</p>
    </section>
  </section>
</template>

<script setup>
defineProps({
  items: {
    type: Array,
    required: true,
  },
})

defineEmits(['clear'])

function confidencePercent(value) {
  const numeric = Number(value || 0)
  return Math.round(numeric <= 1 ? numeric * 100 : numeric)
}
</script>
