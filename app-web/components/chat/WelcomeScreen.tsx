'use client';

import React from 'react';

export default function WelcomeScreen() {
  return (
    <div className="flex flex-1 flex-col items-center justify-center text-center px-6">
      <div className="mb-4 text-5xl">✦</div>
      <h1 className="text-2xl font-semibold text-zinc-800 dark:text-zinc-100 mb-2">
        How can I help you today?
      </h1>
      <p className="text-zinc-500 dark:text-zinc-400 text-sm max-w-sm">
        Start a new conversation by typing a message below.
      </p>
    </div>
  );
}
