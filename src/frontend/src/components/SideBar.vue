<template>
  <aside class="sidebar glass-panel" aria-label="主导航">
    <button
      v-for="item in items"
      :key="item.id"
      class="nav-item"
      :class="{ active: active === item.id }"
      type="button"
      @click="$emit('update:active', item.id)"
    >
      <span class="nav-icon" v-html="item.icon"></span>
      <span class="nav-label">{{ item.label }}</span>
    </button>
  </aside>
</template>

<script setup>
defineProps({
  active: {
    type: String,
    required: true,
  },
})

defineEmits(['update:active'])

const items = [
  {
    id: 'upload',
    label: '上传',
    icon: '<svg viewBox="0 0 24 24"><path d="M12 16V4m0 0 4.5 4.5M12 4 7.5 8.5"/><path d="M20 15.5a4.5 4.5 0 0 0-4.4-4.5 6.5 6.5 0 0 0-12.1 3A4 4 0 0 0 4 22h14a4 4 0 0 0 2-7.5Z"/></svg>',
  },
  {
    id: 'record',
    label: '录音',
    icon: '<svg viewBox="0 0 24 24"><path d="M12 14a4 4 0 0 0 4-4V6a4 4 0 1 0-8 0v4a4 4 0 0 0 4 4Z"/><path d="M19 10a7 7 0 0 1-14 0m7 7v4m-4 0h8"/></svg>',
  },
  {
    id: 'compare',
    label: '对比',
    icon: '<svg viewBox="0 0 24 24"><path d="M6 5v14M18 5v14"/><path d="m3 8 3-3 3 3M15 16l3 3 3-3"/><path d="M9 12h6"/></svg>',
  },
  {
    id: 'history',
    label: '历史',
    icon: '<svg viewBox="0 0 24 24"><path d="M3 12a9 9 0 1 0 3-6.7"/><path d="M3 4v5h5"/><path d="M12 7v6l4 2"/></svg>',
  },
  {
    id: 'settings',
    label: '设置',
    icon: '<svg viewBox="0 0 24 24"><path d="M12 15.5a3.5 3.5 0 1 0 0-7 3.5 3.5 0 0 0 0 7Z"/><path d="M19.4 15a1.8 1.8 0 0 0 .36 1.98l.05.05a2.1 2.1 0 0 1-2.97 2.97l-.05-.05a1.8 1.8 0 0 0-1.98-.36 1.8 1.8 0 0 0-1.09 1.65V21a2.1 2.1 0 0 1-4.2 0v-.08A1.8 1.8 0 0 0 8.4 19.3a1.8 1.8 0 0 0-1.98.36l-.05.05a2.1 2.1 0 1 1-2.97-2.97l.05-.05A1.8 1.8 0 0 0 3.82 15a1.8 1.8 0 0 0-1.65-1.09H2a2.1 2.1 0 0 1 0-4.2h.08A1.8 1.8 0 0 0 3.7 8.6a1.8 1.8 0 0 0-.36-1.98l-.05-.05A2.1 2.1 0 1 1 6.26 3.6l.05.05A1.8 1.8 0 0 0 8.3 4a1.8 1.8 0 0 0 1.09-1.65V2a2.1 2.1 0 0 1 4.2 0v.08A1.8 1.8 0 0 0 15 3.7a1.8 1.8 0 0 0 1.98-.36l.05-.05A2.1 2.1 0 1 1 20 6.26l-.05.05A1.8 1.8 0 0 0 19.6 8.3a1.8 1.8 0 0 0 1.65 1.09H22a2.1 2.1 0 0 1 0 4.2h-.08A1.8 1.8 0 0 0 19.4 15Z"/></svg>',
  },
]
</script>
