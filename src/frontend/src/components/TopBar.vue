<template>
  <header class="topbar">
    <div class="brand">
      <span class="bird-mark" aria-hidden="true">♬</span>
      <div>
        <p>BirdVoice</p>
        <span>森林鸟鸣识别工作台</span>
      </div>
    </div>

    <label class="model-picker">
      <span>模型</span>
      <select :value="model" :disabled="!backendReady || !models.length" @change="$emit('update:model', $event.target.value)">
        <option v-if="!models.length" value="">等待真实模型</option>
        <option v-for="item in models" :key="item.value || item" :value="item.value || item">{{ item.label || item }}</option>
      </select>
    </label>
  </header>
</template>

<script setup>
defineProps({
  model: {
    type: String,
    required: true,
  },
  models: {
    type: Array,
    required: true,
  },
  backendReady: {
    type: Boolean,
    default: true,
  },
})

defineEmits(['update:model'])
</script>
