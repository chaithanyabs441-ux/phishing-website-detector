'use strict';

document.addEventListener('DOMContentLoaded', () => {
  const form = document.getElementById('scan-form');
  const urlInput = document.getElementById('url');
  const fileInput = document.getElementById('html_file');
  const fileLabel = document.getElementById('file-label');
  const fileSubtitle = document.getElementById('file-subtitle');
  const fileActions = document.getElementById('selected-file-actions');
  const fileSize = document.getElementById('file-size');
  const fileError = document.getElementById('file-error');
  const urlError = document.getElementById('url-error');
  const options = document.getElementById('html-options');
  const zone = document.getElementById('upload-zone');
  const submitButton = document.getElementById('scan-button');
  const buttonLabel = submitButton.querySelector('.button-label');
  const spinner = submitButton.querySelector('.button-spinner');
  const status = document.getElementById('form-status');

  const showError = (node, message, input) => {
    node.textContent = message;
    node.hidden = !message;
    input.setAttribute('aria-invalid', message ? 'true' : 'false');
  };
  const updateFile = () => {
    const file = fileInput.files[0];
    let message = '';
    if (file && !/\.html?$/i.test(file.name)) message = 'Choose a file ending in .html or .htm.';
    if (file && file.size > 10000000) message = 'Choose an HTML file smaller than 10 MB.';
    showError(fileError, message, fileInput);
    fileLabel.textContent = file ? file.name : 'Choose an HTML file';
    fileSubtitle.textContent = file ? 'Saved page selected for structural analysis' : 'or drop a saved page here · .html / .htm · up to 10 MB';
    fileActions.hidden = !file;
    fileSize.textContent = file ? (file.size >= 1000000 ? (file.size / 1000000).toFixed(2) + ' MB' : Math.max(1, Math.round(file.size / 1000)) + ' KB') : '';
    if (file) options.open = true;
    return !message;
  };
  fileInput.addEventListener('change', updateFile);
  document.getElementById('remove-file').addEventListener('click', () => {
    fileInput.value = '';
    updateFile();
    fileInput.focus();
  });
  ['dragenter', 'dragover'].forEach(type => zone.addEventListener(type, event => {
    event.preventDefault();
    zone.classList.add('is-dragging');
  }));
  ['dragleave', 'drop'].forEach(type => zone.addEventListener(type, event => {
    event.preventDefault();
    zone.classList.remove('is-dragging');
  }));
  zone.addEventListener('drop', event => {
    if (!event.dataTransfer.files.length) return;
    if (event.dataTransfer.files.length > 1) {
      showError(fileError, 'Choose one HTML file for this analysis.', fileInput);
      return;
    }
    fileInput.files = event.dataTransfer.files;
    updateFile();
  });

  form.noValidate = true;
  form.addEventListener('submit', event => {
    const url = urlInput.value.trim();
    showError(urlError, url ? '' : 'Enter the website URL you want to check.', urlInput);
    const validFile = updateFile();
    if (!url || !validFile || form.getAttribute('aria-busy') === 'true') {
      event.preventDefault();
      if (!url) urlInput.focus();
      else if (!validFile) { options.open = true; fileInput.focus(); }
      return;
    }
    form.setAttribute('aria-busy', 'true');
    submitButton.disabled = true;
    buttonLabel.textContent = 'Analyzing URL';
    spinner.hidden = false;
    status.textContent = fileInput.files.length ? 'Analyzing URL + HTML…' : 'Analyzing URL…';
  });
  window.addEventListener('pageshow', () => {
    form.removeAttribute('aria-busy');
    submitButton.disabled = false;
    buttonLabel.textContent = 'Analyze URL';
    spinner.hidden = true;
    status.textContent = 'Ready when you are.';
  });

  const engine = document.getElementById('engine-state');
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 5000);
  fetch('/health', { signal: controller.signal, credentials: 'same-origin' })
    .then(response => { if (!response.ok) throw new Error('Engine unavailable'); return response.json(); })
    .then(data => {
      engine.dataset.state = data.status === 'ok' ? 'ready' : 'error';
      document.getElementById('engine-label').textContent = data.status === 'ok' ? 'Engine ready' : 'Engine unavailable';
    })
    .catch(() => { engine.dataset.state = 'error'; document.getElementById('engine-label').textContent = 'Engine unavailable'; })
    .finally(() => clearTimeout(timeout));

  const copy = document.getElementById('copy-url');
  if (copy) copy.addEventListener('click', async () => {
    try {
      await navigator.clipboard.writeText(document.querySelector('.scanned-url').textContent.trim());
      copy.textContent = 'URL copied';
      setTimeout(() => { copy.textContent = 'Copy URL'; }, 1800);
    } catch {
      copy.textContent = 'Select URL to copy';
    }
  });
  const download = document.getElementById('download-report');
  if (download) download.addEventListener('click', () => {
    const result = JSON.parse(document.getElementById('scan-data').textContent);
    const blob = new Blob([JSON.stringify(result, null, 2)], { type: 'application/json' });
    const objectUrl = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = objectUrl;
    link.download = 'phishreaper-report.json';
    document.body.append(link);
    link.click();
    link.remove();
    setTimeout(() => URL.revokeObjectURL(objectUrl), 1000);
  });
  document.querySelectorAll('a[href="#url"]').forEach(link => link.addEventListener('click', () => {
    urlInput.focus();
    urlInput.select();
  }));
  const mobileNav = document.getElementById('mobile-nav');
  const mobileToggle = mobileNav.querySelector('summary');
  mobileNav.addEventListener('toggle', () => {
    mobileToggle.setAttribute('aria-expanded', String(mobileNav.open));
    mobileToggle.setAttribute('aria-label', mobileNav.open ? 'Close navigation' : 'Open navigation');
  });
  mobileNav.querySelectorAll('a').forEach(link => link.addEventListener('click', () => { mobileNav.open = false; }));
  mobileNav.addEventListener('focusout', event => {
    if (mobileNav.open && !mobileNav.contains(event.relatedTarget)) mobileNav.open = false;
  });
  mobileNav.addEventListener('keydown', event => {
    if (event.key === 'Escape') { mobileNav.open = false; mobileToggle.focus(); }
  });
  const results = document.getElementById('results');
  if (results) {
    results.focus({ preventScroll: true });
    results.scrollIntoView({ behavior: 'auto', block: 'start' });
  }
});
