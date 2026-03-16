'use client';

import React, { useMemo } from 'react';
import MarkdownIt from 'markdown-it';
import hljs from 'highlight.js';
import 'highlight.js/styles/atom-one-light.css';

interface Props {
  content: string;
}

export default function MarkdownRenderer({ content }: Props) {
  const html = useMemo(() => {
    const md = new MarkdownIt({
      html: false,
      linkify: true,
      typographer: true,
      highlight: (code, lang) => {
        if (lang && hljs.getLanguage(lang)) {
          try {
            return hljs.highlight(code, { language: lang }).value;
          } catch (err) {
            return code;
          }
        }
        return code;
      },
    });

    return md.render(content);
  }, [content]);

  return (
    <div
      className="prose max-w-none text-zinc-900 dark:text-zinc-100
        [&_pre]:!bg-[#f9f9f9] [&_pre]:!text-zinc-900 [&_pre]:!p-4 [&_pre]:!rounded-lg [&_pre]:!border [&_pre]:!border-zinc-300 [&_pre]:!overflow-x-auto
        dark:[&_pre]:!bg-zinc-800 dark:[&_pre]:!text-zinc-100 dark:[&_pre]:!border-zinc-700
        [&_code]:!bg-[#f9f9f9] [&_code]:!text-zinc-900 [&_code]:!px-1.5 [&_code]:!py-0.5 [&_code]:!rounded [&_code]:!text-sm [&_code]:!font-mono
        dark:[&_code]:!bg-zinc-700 dark:[&_code]:!text-zinc-100
        [&_h1]:!text-2xl [&_h1]:!font-bold [&_h1]:!mt-4 [&_h1]:!mb-2
        [&_h2]:!text-xl [&_h2]:!font-bold [&_h2]:!mt-3 [&_h2]:!mb-2
        [&_h3]:!text-lg [&_h3]:!font-bold [&_h3]:!mt-2 [&_h3]:!mb-1
        [&_p]:!my-2 [&_p]:!leading-relaxed
        [&_ul]:!list-disc [&_ul]:!list-inside [&_ul]:!space-y-1 [&_ul]:!my-2 [&_ul]:!ml-2
        [&_ol]:!list-decimal [&_ol]:!list-inside [&_ol]:!space-y-1 [&_ol]:!my-2 [&_ol]:!ml-2
        [&_li]:!ml-2
        [&_a]:!text-blue-500 [&_a]:!underline hover:[&_a]:!text-blue-600
        [&_blockquote]:!border-l-4 [&_blockquote]:!border-zinc-300 [&_blockquote]:!pl-4 [&_blockquote]:!italic [&_blockquote]:!text-zinc-600 [&_blockquote]:!my-2
        dark:[&_blockquote]:!border-zinc-600 dark:[&_blockquote]:!text-zinc-400
        [&_table]:!border-collapse [&_table]:!border [&_table]:!border-zinc-300 [&_table]:!my-2 [&_table]:!text-sm
        dark:[&_table]:!border-zinc-600
        [&_thead]:!bg-zinc-200 dark:[&_thead]:!bg-zinc-800
        [&_th]:!border [&_th]:!border-zinc-300 [&_th]:!px-3 [&_th]:!py-2 [&_th]:!font-bold [&_th]:!text-left
        dark:[&_th]:!border-zinc-600
        [&_td]:!border [&_td]:!border-zinc-300 [&_td]:!px-3 [&_td]:!py-2
        dark:[&_td]:!border-zinc-600
        [&_strong]:!font-bold
        [&_em]:!italic
        [&_hr]:!my-4 [&_hr]:!border-zinc-600
      "
      dangerouslySetInnerHTML={{ __html: html }}
    />
  );
}
