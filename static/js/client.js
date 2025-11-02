// Client-side logic for UI tabs, snapping, uploads, info panel

const tabLive = document.getElementById('tabLive');
const tabUpload = document.getElementById('tabUpload');
const liveTab = document.getElementById('liveTab');
const uploadTab = document.getElementById('uploadTab');
const videoFeed = document.getElementById('videoFeed');
const snapBtn = document.getElementById('snapBtn');
const liveResult = document.getElementById('liveResult');
const uploadBtn = document.getElementById('uploadBtn');
const uploadFile = document.getElementById('uploadFile');
const uploadResult = document.getElementById('uploadResult');
const toggleInfo = document.getElementById('toggleInfo');
const infoPanel = document.getElementById('infoPanel');

// Tabs
tabLive.onclick = () => {
  liveTab.classList.remove('hidden');
  uploadTab.classList.add('hidden');
  tabLive.classList.add('bg-blue-600','text-white');
  tabUpload.classList.remove('bg-blue-600','text-white');
};
tabUpload.onclick = () => {
  uploadTab.classList.remove('hidden');
  liveTab.classList.add('hidden');
  tabUpload.classList.add('bg-blue-600','text-white');
  tabLive.classList.remove('bg-blue-600','text-white');
};

// Toggle info panel
toggleInfo.onclick = () => {
  infoPanel.style.maxHeight = infoPanel.style.maxHeight === '0px' ? '200px' : '0px';
};

// Snap button
snapBtn.onclick = async () => {
  const endpoint = isMobile ? '/snap_mobile' : '/snap_live';
  const res = await fetch(endpoint, { method: 'POST' });
  const data = await res.json();
  if (data.success) {
    liveResult.innerHTML = `<p>✅ Saved</p>
      <img src="${data.file}?t=${Date.now()}" class="rounded mt-2 w-full"/>
      <a href="${data.file}" download class="text-blue-600 underline">📥 Download</a>`;
  } else {
    liveResult.innerHTML = `<p class="text-red-600">${data.error}</p>`;
  }
};

// Upload
uploadBtn.onclick = async () => {
  const file = uploadFile.files[0];
  if (!file) { alert("Please choose an image first."); return; }
  uploadResult.innerHTML = "Detecting...";
  const form = new FormData();
  form.append('file', file);
  const res = await fetch('/upload', { method: 'POST', body: form });
  const data = await res.json();
  if (data.success) {
    uploadResult.innerHTML = `<img src="${data.file}?t=${Date.now()}" class="rounded w-full"/>
      <a href="${data.file}" download class="text-blue-600 underline">📥 Download</a>`;
  } else {
    uploadResult.innerHTML = `<p class="text-red-600">❌ ${data.error}</p>`;
  }
};
