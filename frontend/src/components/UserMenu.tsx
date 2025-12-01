"use client";

import { useState, useRef, useEffect } from 'react';
import { useAuth } from '@/contexts/AuthContext';
import { AuthModal } from './AuthModal';
import { User, LogOut, Cloud, ChevronDown } from 'lucide-react';

interface UserMenuProps {
    onOpenCloudProjects: () => void;
}

export function UserMenu({ onOpenCloudProjects }: UserMenuProps) {
    const { user, signOut } = useAuth();
    const [showAuthModal, setShowAuthModal] = useState(false);
    const [isOpen, setIsOpen] = useState(false);
    const menuRef = useRef<HTMLDivElement>(null);
    const [dropdownPosition, setDropdownPosition] = useState({ top: 0, right: 0 });

    useEffect(() => {
        const handleClickOutside = (event: MouseEvent) => {
            if (menuRef.current && !menuRef.current.contains(event.target as Node)) {
                setIsOpen(false);
            }
        };

        const updatePosition = () => {
            if (menuRef.current && isOpen) {
                const rect = menuRef.current.getBoundingClientRect();
                setDropdownPosition({
                    top: rect.bottom + 8,
                    right: window.innerWidth - rect.right
                });
            }
        };

        document.addEventListener('mousedown', handleClickOutside);
        window.addEventListener('resize', updatePosition);
        window.addEventListener('scroll', updatePosition, true);

        if (isOpen) {
            updatePosition();
        }

        return () => {
            document.removeEventListener('mousedown', handleClickOutside);
            window.removeEventListener('resize', updatePosition);
            window.removeEventListener('scroll', updatePosition, true);
        };
    }, [isOpen]);

    if (!user) {
        return (
            <>
                <button
                    onClick={() => setShowAuthModal(true)}
                    className="flex items-center gap-2 px-3 py-2 text-sm font-medium text-gray-300 hover:text-white hover:bg-[#2a2a2a] rounded-lg transition-colors"
                >
                    <User size={16} />
                    Login
                </button>
                <AuthModal isOpen={showAuthModal} onClose={() => setShowAuthModal(false)} />
            </>
        );
    }

    return (
        <div className="relative" ref={menuRef}>
            <button
                onClick={() => setIsOpen(!isOpen)}
                className="flex items-center gap-2 px-3 py-2 text-sm font-medium text-gray-300 hover:text-white hover:bg-[#2a2a2a] rounded-lg transition-colors"
            >
                <div className="w-6 h-6 rounded-full bg-blue-600 flex items-center justify-center text-xs text-white font-bold">
                    {user.email?.[0].toUpperCase()}
                </div>
                <span className="max-w-[100px] truncate">{user.email}</span>
                <ChevronDown size={14} className={`transition-transform ${isOpen ? 'rotate-180' : ''}`} />
            </button>

            {isOpen && (
                <div
                    className="fixed w-48 bg-[#252525] border border-gray-700 rounded-lg shadow-xl overflow-hidden z-50"
                    style={{ top: dropdownPosition.top, right: dropdownPosition.right }}
                >
                    <div className="py-1">
                        <button
                            onClick={() => {
                                onOpenCloudProjects();
                                setIsOpen(false);
                            }}
                            className="w-full px-4 py-2 text-left text-sm text-gray-300 hover:bg-[#2a2a2a] hover:text-white flex items-center gap-2"
                        >
                            <Cloud size={16} className="text-blue-400" />
                            My Projects
                        </button>
                        <div className="h-px bg-gray-700 my-1" />
                        <button
                            onClick={() => {
                                signOut();
                                setIsOpen(false);
                            }}
                            className="w-full px-4 py-2 text-left text-sm text-gray-300 hover:bg-[#2a2a2a] hover:text-white flex items-center gap-2"
                        >
                            <LogOut size={16} className="text-red-400" />
                            Sign Out
                        </button>
                    </div>
                </div>
            )}
        </div>
    );
}
