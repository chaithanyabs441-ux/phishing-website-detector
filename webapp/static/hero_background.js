import fx from './taste/effects/fx.js';

const grounds = await fx.mountAll(document);
requestAnimationFrame(() => grounds.forEach(ground => ground?.pause()));
